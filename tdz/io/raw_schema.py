"""Raw input data schemas for the ADS-B timeseries and QAR truth sources.

These describe the *external, on-disk* column layouts exactly as delivered by
the data providers — they are deliberately separate from the pipeline-internal
models in :mod:`tdz.models` (``FlightRecord``, ``QARTruthRecord``). The ingest
module (:mod:`tdz.io`, Task 9) parses these raw rows, validates them, and
converts every quantity to SI (meters, m/s, seconds, radians) before building
the internal records (Units Convention).

Units convention (raw vs internal)
-----------------------------------
Raw provider columns keep the provider's native names and units (knots, feet,
ft/min, degrees) and therefore do NOT carry SI unit suffixes. The explicit
``_m`` / ``_mps`` / ``_s`` suffixes only appear on the internal models after
ingest conversion. Where a raw unit is genuinely ambiguous in ADS-B/QAR feeds
(runway length, geometric height, deceleration), the :class:`FieldSpec` ``unit``
is annotated with ``TODO: confirm`` so the conversion can be pinned down against
the real files rather than guessed silently.

Field constancy
---------------
ADS-B rows are a per-sample timeseries, but several columns are per-flight
constants repeated on every row (airport, runway geometry, registration, type,
landing summary). :attr:`FieldSpec.constancy` distinguishes ``"per_sample"``
(varies row to row) from ``"per_flight"`` (constant within a flight_id) to guide
ingest (the per-flight columns should be deduplicated/validated for consistency,
not treated as independent observations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "FieldSpec",
    "ADSB_FIELDS",
    "QAR_FIELDS",
    "ADSBRecord",
    "QARRecord",
    "adsb_columns",
    "qar_columns",
    "adsb_pandas_dtypes",
    "qar_pandas_dtypes",
    "ADSB_TO_FLIGHTRECORD",
    "QAR_TO_TRUTHRECORD",
    "JOIN_KEYS",
]


# ---------------------------------------------------------------------------
# Field metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """Metadata describing one raw input column.

    Attributes:
        name: Exact column name as delivered by the provider.
        dtype: Pandas/numpy-friendly dtype identifier ("float64", "string",
            "boolean", "Int64"). Nullable dtypes use pandas extension types.
        unit: Native unit of the raw value (free text; "TODO: confirm" where
            the provider unit is ambiguous). "—" for unitless ids/labels.
        nullable: Whether the column may be missing/NaN on a given row.
        constancy: "per_sample" (varies per timeseries row) or "per_flight"
            (constant within a flight_id; repeated on every row).
        description: Human-readable meaning and any ingest notes.
    """

    name: str
    dtype: str
    unit: str
    nullable: bool
    constancy: str
    description: str


# ---------------------------------------------------------------------------
# ADS-B timeseries schema
# ---------------------------------------------------------------------------

#: Raw ADS-B timeseries columns, in delivery order.
ADSB_FIELDS: list[FieldSpec] = [
    FieldSpec("flight_id", "string", "—", False, "per_flight",
              "Unique identifier for the flight/landing event (join key within ADS-B)."),
    FieldSpec("timestamp", "float64", "epoch seconds (TODO: confirm vs ISO8601)", False, "per_sample",
              "Sample/message time for this row."),
    FieldSpec("hexid", "string", "—", False, "per_flight",
              "24-bit ICAO Mode S address (hex). Join key to QAR.HEXID."),
    FieldSpec("time_position", "float64", "epoch seconds", True, "per_sample",
              "Emission time of the position fix; preserved separately from time_velocity "
              "for async sources (do not merge into a single sample time)."),
    FieldSpec("time_velocity", "float64", "epoch seconds", True, "per_sample",
              "Emission time of the velocity fix; preserved separately from time_position."),
    FieldSpec("latitude", "float64", "degrees", True, "per_sample",
              "WGS-84 latitude of the position fix."),
    FieldSpec("longitude", "float64", "degrees", True, "per_sample",
              "WGS-84 longitude of the position fix."),
    FieldSpec("ground_speed", "float64", "knots", True, "per_sample",
              "Ground speed (combined). See ground_speed_airborne/surface for phase-specific values."),
    FieldSpec("airport", "string", "—", False, "per_flight",
              "Destination airport identifier (ICAO)."),
    FieldSpec("runway", "string", "—", False, "per_flight",
              "Landing runway designator (e.g. '04L')."),
    FieldSpec("on_ground", "boolean", "—", True, "per_sample",
              "ADS-B air/ground flag. Upper time-bound only (delayed transition); never t_td."),
    FieldSpec("landing_timestamp", "float64", "epoch seconds", True, "per_flight",
              "Provider-derived landing time for the flight (repeated per row). Not ground truth."),
    FieldSpec("landing_latitude", "float64", "degrees", True, "per_flight",
              "Provider-derived landing latitude (repeated per row). Not ground truth."),
    FieldSpec("landing_longitude", "float64", "degrees", True, "per_flight",
              "Provider-derived landing longitude (repeated per row). Not ground truth."),
    FieldSpec("aircraft_registration", "string", "—", False, "per_flight",
              "Tail number. Grouping key for the tail-grouped validation split."),
    FieldSpec("runway_length", "float64", "meters (TODO: confirm m vs ft)", True, "per_flight",
              "Landing runway length."),
    FieldSpec("model_icao", "string", "—", False, "per_flight",
              "ICAO aircraft type designator (e.g. 'B738'). Lever-arm lookup key."),
    FieldSpec("aircraft_model_number_and_subtype", "string", "—", True, "per_flight",
              "Manufacturer model number and subtype label."),
    FieldSpec("threshold_latitude", "float64", "degrees", True, "per_flight",
              "Landing-threshold latitude (origin for along-runway distance)."),
    FieldSpec("threshold_longitude", "float64", "degrees", True, "per_flight",
              "Landing-threshold longitude (origin for along-runway distance)."),
    # NOTE: source column name is 'threshold_diplacement_length' [sic — 'displacement'].
    # Kept verbatim so the schema matches the delivered files; see DISPLACEMENT_TYPO_ALIAS.
    FieldSpec("threshold_diplacement_length", "float64", "meters (TODO: confirm m vs ft)", True, "per_flight",
              "Displaced-threshold distance from the physical runway start at the landing end "
              "[sic: source spells 'diplacement']."),
    FieldSpec("opposite_displacement_length", "float64", "meters (TODO: confirm m vs ft)", True, "per_flight",
              "Displaced-threshold distance at the opposite runway end."),
    FieldSpec("lda_day", "float64", "meters (TODO: confirm m vs ft)", True, "per_flight",
              "Landing Distance Available (day)."),
    FieldSpec("ground_speed_airborne", "float64", "knots", True, "per_sample",
              "Ground speed reported while airborne."),
    FieldSpec("ground_speed_surface", "float64", "knots", True, "per_sample",
              "Ground speed reported while on the surface."),
    FieldSpec("barometric_altitude", "float64", "feet (TODO: confirm)", True, "per_sample",
              "Pressure altitude (QNH-sensitive near the surface). Not used for the geometric crossing."),
    FieldSpec("barometric_vertical_rate", "float64", "ft/min", True, "per_sample",
              "Barometric vertical rate (often null/missing)."),
    FieldSpec("geometric_height", "float64", "feet (TODO: confirm ft vs m); HAE", True, "per_sample",
              "GNSS geometric height above the WGS-84 ellipsoid (HAE). Used for the vertical crossing."),
    FieldSpec("selected_altitude", "float64", "feet (TODO: confirm)", True, "per_sample",
              "MCP/FMS selected altitude (autopilot target)."),
    FieldSpec("track", "float64", "degrees true", True, "per_sample",
              "Track angle over the ground."),
]


# ---------------------------------------------------------------------------
# QAR truth schema
# ---------------------------------------------------------------------------

#: Raw QAR (Quick Access Recorder) ground-truth columns, in delivery order.
QAR_FIELDS: list[FieldSpec] = [
    FieldSpec("registration", "string", "—", False, "per_flight",
              "Tail number. Join key to ADS-B aircraft_registration."),
    FieldSpec("landing_time", "float64", "epoch seconds (TODO: confirm clock/zone)", False, "per_flight",
              "QAR touchdown timestamp (ground truth). On the QAR clock; needs ADS-B clock alignment "
              "before use as a time-domain label."),
    FieldSpec("destination_actual_icao", "string", "—", False, "per_flight",
              "Actual destination airport (ICAO)."),
    FieldSpec("destination_actual_runway", "string", "—", False, "per_flight",
              "Actual landing runway designator."),
    FieldSpec("destination_runway_length", "float64", "meters (TODO: confirm m vs ft)", True, "per_flight",
              "Runway length at the actual destination runway."),
    FieldSpec("longitude_at_touchdown", "float64", "degrees", False, "per_flight",
              "QAR touchdown longitude. Source of clock-independent along-runway distance truth."),
    FieldSpec("latitude_at_touchdown", "float64", "degrees", False, "per_flight",
              "QAR touchdown latitude. Source of clock-independent along-runway distance truth."),
    FieldSpec("destination_runway_threshold_latitude", "float64", "degrees", True, "per_flight",
              "Threshold latitude of the actual landing runway."),
    FieldSpec("destination_runway_threshold_longitude", "float64", "degrees", True, "per_flight",
              "Threshold longitude of the actual landing runway."),
    FieldSpec("deceleration_at_60kt", "float64", "m/s^2 (TODO: confirm units)", True, "per_flight",
              "Deceleration at 60 kt ground speed during rollout."),
    FieldSpec("cas_touchdown", "float64", "knots", True, "per_flight",
              "Calibrated airspeed at touchdown."),
    FieldSpec("gs_touchdown", "float64", "knots", True, "per_flight",
              "Ground speed at touchdown (truth for the touchdown-speed metric)."),
    FieldSpec("aircraft_type", "string", "—", False, "per_flight",
              "Aircraft type (manufacturer/family or ICAO type — TODO: confirm coding vs ADS-B model_icao)."),
    FieldSpec("aircraft_subtype", "string", "—", True, "per_flight",
              "Aircraft subtype label."),
    FieldSpec("runway_length", "float64", "meters (TODO: confirm; relation to destination_runway_length)", True, "per_flight",
              "Runway length (appears alongside destination_runway_length; confirm which is authoritative)."),
    FieldSpec("HEXID", "string", "—", True, "per_flight",
              "24-bit ICAO Mode S address (hex). Join key to ADS-B hexid."),
]


#: The source ADS-B column spells displacement as 'diplacement'. Use this alias
#: constant rather than re-typing the misspelling throughout the codebase.
DISPLACEMENT_TYPO_ALIAS: str = "threshold_diplacement_length"


# ---------------------------------------------------------------------------
# Typed raw-row dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ADSBRecord:
    """One raw ADS-B timeseries row, as delivered (pre-conversion).

    Field names and units match :data:`ADSB_FIELDS` exactly. Optional fields may
    be missing/NaN on a given row. Ingest (Task 9) converts to SI and builds the
    internal :class:`tdz.models.FlightRecord` (one record aggregates many rows).
    """

    # Identity / per-flight constants
    flight_id: str
    hexid: str
    airport: str
    runway: str
    aircraft_registration: str
    model_icao: str

    # Per-sample dynamic
    timestamp: float
    time_position: Optional[float] = None
    time_velocity: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    ground_speed: Optional[float] = None
    on_ground: Optional[bool] = None
    ground_speed_airborne: Optional[float] = None
    ground_speed_surface: Optional[float] = None
    barometric_altitude: Optional[float] = None
    barometric_vertical_rate: Optional[float] = None
    geometric_height: Optional[float] = None
    selected_altitude: Optional[float] = None
    track: Optional[float] = None

    # Per-flight summary / geometry (constant within flight_id)
    landing_timestamp: Optional[float] = None
    landing_latitude: Optional[float] = None
    landing_longitude: Optional[float] = None
    runway_length: Optional[float] = None
    aircraft_model_number_and_subtype: Optional[str] = None
    threshold_latitude: Optional[float] = None
    threshold_longitude: Optional[float] = None
    # [sic] source spells 'diplacement'; kept verbatim to match the files.
    threshold_diplacement_length: Optional[float] = None
    opposite_displacement_length: Optional[float] = None
    lda_day: Optional[float] = None


@dataclass
class QARRecord:
    """One raw QAR ground-truth row, as delivered (pre-conversion).

    Field names and units match :data:`QAR_FIELDS` exactly. Ingest converts to
    SI and builds the internal :class:`tdz.models.QARTruthRecord`.
    """

    registration: str
    landing_time: float
    destination_actual_icao: str
    destination_actual_runway: str
    longitude_at_touchdown: float
    latitude_at_touchdown: float
    aircraft_type: str
    destination_runway_length: Optional[float] = None
    destination_runway_threshold_latitude: Optional[float] = None
    destination_runway_threshold_longitude: Optional[float] = None
    deceleration_at_60kt: Optional[float] = None
    cas_touchdown: Optional[float] = None
    gs_touchdown: Optional[float] = None
    aircraft_subtype: Optional[str] = None
    runway_length: Optional[float] = None
    HEXID: Optional[str] = None


# ---------------------------------------------------------------------------
# Column / dtype helpers (for loading CSV / Parquet)
# ---------------------------------------------------------------------------


def adsb_columns() -> list[str]:
    """Ordered list of raw ADS-B column names."""
    return [f.name for f in ADSB_FIELDS]


def qar_columns() -> list[str]:
    """Ordered list of raw QAR column names."""
    return [f.name for f in QAR_FIELDS]


def adsb_pandas_dtypes() -> dict[str, str]:
    """Mapping of ADS-B column -> pandas dtype, suitable for ``read_csv(dtype=...)``."""
    return {f.name: f.dtype for f in ADSB_FIELDS}


def qar_pandas_dtypes() -> dict[str, str]:
    """Mapping of QAR column -> pandas dtype, suitable for ``read_csv(dtype=...)``."""
    return {f.name: f.dtype for f in QAR_FIELDS}


# ---------------------------------------------------------------------------
# Mappings to the internal models (documentation + ingest guidance)
# ---------------------------------------------------------------------------

#: Raw ADS-B column -> internal FlightRecord field. Columns absent here are
#: either aggregated across rows, converted, or carried only in diagnostics.
#: Conversions (knots->m/s, feet->m, etc.) happen in ingest, not in this map.
ADSB_TO_FLIGHTRECORD: dict[str, str] = {
    "flight_id": "flight_id",
    "model_icao": "aircraft_type",
    "time_position": "position_times",
    "time_velocity": "velocity_times",
    "latitude": "latitudes",
    "longitude": "longitudes",
    "geometric_height": "geometric_altitudes",      # ft (HAE) -> m
    "barometric_altitude": "barometric_altitudes",   # ft -> m
    "ground_speed": "groundspeeds",                  # stays knots on FlightRecord
    "track": "tracks",
    "barometric_vertical_rate": "baro_vertical_rates",
    "on_ground": "on_ground_flags",
    # runway geometry -> FlightRecord.runway (RunwayReference):
    # threshold_latitude/longitude, runway_length, threshold_diplacement_length.
}

#: Raw QAR column -> internal QARTruthRecord field.
QAR_TO_TRUTHRECORD: dict[str, str] = {
    "landing_time": "touchdown_time_qar",
    "latitude_at_touchdown": "touchdown_lat",
    "longitude_at_touchdown": "touchdown_lon",
    "aircraft_type": "aircraft_type",
    "destination_actual_runway": "runway_id",
    "destination_actual_icao": "airport_id",
    "registration": "tail_number",
}

#: Keys used to join ADS-B samples to QAR truth. HEXID/hexid is the primary
#: airframe join; registration <-> aircraft_registration is the secondary check.
JOIN_KEYS: dict[str, tuple[str, str]] = {
    "airframe": ("hexid", "HEXID"),
    "registration": ("aircraft_registration", "registration"),
}
