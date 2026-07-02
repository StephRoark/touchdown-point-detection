"""File readers: delivered ADS-B / QAR data files -> pipeline-internal records.

This is the missing link between the on-disk files described by
:mod:`tdz.io.raw_schema` and the pipeline-internal models in :mod:`tdz.models`.
It reads the delivered ADS-B timeseries and QAR truth files (CSV or Parquet),
validates their columns against the raw schema, groups samples into flights,
converts units at this single boundary, and emits :class:`~tdz.models.FlightRecord`
and :class:`~tdz.models.QARTruthRecord` objects ready for the existing pipeline.

What the delivered files do NOT contain
---------------------------------------
The ADS-B file carries the landing threshold lat/long and the runway length,
but a :class:`~tdz.models.RunwayReference` also needs the runway **heading**,
**threshold elevation (with datum)**, and **width** — none of which are in the
delivered columns. Those come from a **runway supplement table**
(:func:`read_runway_supplement`): one row per (airport, runway) supplying
``heading_deg``, ``elevation_m`` + ``elevation_datum``, ``width_m``, and
optionally ``geoid_undulation_m`` and fallback threshold coordinates / length.
Flights whose (airport, runway) is absent from the supplement are skipped with
an explicit reason — never guessed.

Unit assumptions are explicit
------------------------------
Several raw units are marked "TODO: confirm" in the schema (README "Status and
open items"). Rather than bury those assumptions, they are collected in one
:class:`RawUnits` object whose defaults mirror the schema annotations
(``geometric_height``/``barometric_altitude`` in feet; runway length /
displacement lengths in meters). The units used are carried on the
:class:`AdsbReadResult` so every downstream artifact can record what was
assumed; when the provider confirms the true units, flip the relevant
:class:`RawUnits` field — no code changes.

Failure policy
--------------
Malformed *files* (missing required columns, unreadable format) raise
:class:`RawFileError` immediately. Malformed *flights* (inconsistent per-flight
constants, missing runway geometry, no valid samples) are skipped with a
machine-readable :class:`SkippedFlight` reason and never abort the batch —
mirroring the pipeline's flag-don't-crash philosophy (Req 14).

Units convention: this module is the raw->SI boundary for file input. Altitude
columns are converted to meters here; groundspeed stays in knots and baro
vertical rate in ft/min on the ``FlightRecord`` (the same native units the
timebase consumes), exactly as :mod:`tdz.io.ingest` does for in-memory sources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from tdz.geo.errors import InvalidRunwayReferenceError
from tdz.geo.runway import validate_runway_reference
from tdz.io.ingest import _on_ground_transition_time, _sort_by_time
from tdz.io.raw_schema import (
    ADSB_FIELDS,
    DISPLACEMENT_TYPO_ALIAS,
    QAR_FIELDS,
    QARRecord,
)
from tdz.models import FlightRecord, QARTruthRecord, RunwayReference

__all__ = [
    "FT_TO_M",
    "RawFileError",
    "RawUnits",
    "RunwaySupplementEntry",
    "SkippedFlight",
    "AdsbFlightMeta",
    "AdsbReadResult",
    "QARMatchResult",
    "read_adsb_file",
    "read_qar_file",
    "read_runway_supplement",
    "flight_records_from_adsb",
    "read_adsb_flights",
    "qar_records_from_dataframe",
    "match_qar_to_flights",
    "CORRECTED_DISPLACEMENT_NAME",
]

#: Feet -> meters (exact).
FT_TO_M = 0.3048

#: The correctly-spelled variant of the source's 'threshold_diplacement_length'
#: [sic] column. Files delivered with the corrected spelling are normalized to
#: the schema (typo) name on read, so the rest of the code sees one name.
CORRECTED_DISPLACEMENT_NAME = "threshold_displacement_length"

#: Pre-alignment placeholder for QARTruthRecord.clock_offset_quality: the
#: offset is estimated by the Task-21 clock-alignment step, not at read time.
_CLOCK_QUALITY_UNALIGNED = ""


class RawFileError(ValueError):
    """A delivered file is unreadable or does not match the raw schema."""


# ---------------------------------------------------------------------------
# Unit assumptions (explicit, flip-able)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawUnits:
    """The unit assumed for each 'TODO: confirm' raw column (see module doc).

    Each field takes ``"ft"`` or ``"m"``. Defaults mirror the raw-schema
    annotations. These are data-description knobs, not tuning parameters: set
    them to whatever the provider confirms.
    """

    geometric_height: str = "ft"        # ADS-B geometric_height (HAE)
    barometric_altitude: str = "ft"     # ADS-B barometric_altitude
    runway_length: str = "m"            # runway_length / destination_runway_length
    displacement_length: str = "m"      # threshold/opposite displacement lengths

    def __post_init__(self) -> None:
        for name in ("geometric_height", "barometric_altitude",
                     "runway_length", "displacement_length"):
            value = getattr(self, name)
            if value not in ("ft", "m"):
                raise ValueError(
                    f"RawUnits.{name} must be 'ft' or 'm', got {value!r}"
                )

    @staticmethod
    def to_meters(value: float, unit: str) -> float:
        """Convert ``value`` in ``unit`` ('ft' | 'm') to meters (NaN-safe)."""
        if unit == "ft":
            return value * FT_TO_M
        return value


# ---------------------------------------------------------------------------
# File loading (CSV / Parquet) + schema validation
# ---------------------------------------------------------------------------


def _load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a CSV / Parquet file into a DataFrame (format by extension)."""
    p = Path(path)
    if not p.exists():
        raise RawFileError(f"input file does not exist: {p}")
    suffixes = "".join(p.suffixes).lower()
    try:
        if suffixes.endswith((".parquet", ".pq")):
            try:
                return pd.read_parquet(p)
            except ImportError as exc:  # pragma: no cover - engine-dependent
                raise RawFileError(
                    f"reading {p.name} requires a parquet engine "
                    "(pip install pyarrow)"
                ) from exc
        if ".csv" in suffixes:  # .csv, .csv.gz, .csv.bz2 ...
            return pd.read_csv(p)
        raise RawFileError(
            f"unsupported input format for {p.name!r}: expected .csv[.gz] or .parquet"
        )
    except RawFileError:
        raise
    except Exception as exc:
        raise RawFileError(f"failed to read {p}: {exc}") from exc


def _coerce_boolean(series: pd.Series) -> pd.Series:
    """Coerce a provider boolean column ('True'/'false'/1/0/NaN) to nullable bool."""
    mapping = {
        "true": True, "false": False, "t": True, "f": False,
        "1": True, "0": False, "yes": True, "no": False,
    }

    def one(v: object) -> object:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return pd.NA
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer, float, np.floating)):
            return bool(v)
        key = str(v).strip().lower()
        if key in mapping:
            return mapping[key]
        return pd.NA

    return series.map(one).astype("boolean")


def _apply_schema_dtypes(df: pd.DataFrame, fields: Iterable) -> pd.DataFrame:
    """Coerce columns to their schema dtypes (tolerant of provider quirks)."""
    out = df.copy()
    for spec in fields:
        if spec.name not in out.columns:
            continue
        if spec.dtype == "float64":
            out[spec.name] = pd.to_numeric(out[spec.name], errors="coerce")
        elif spec.dtype == "boolean":
            out[spec.name] = _coerce_boolean(out[spec.name])
        elif spec.dtype == "string":
            out[spec.name] = out[spec.name].astype("string")
    return out


def _require_columns(df: pd.DataFrame, fields: Iterable, *, label: str) -> None:
    """Raise :class:`RawFileError` when a required (non-nullable) column is absent."""
    missing = [f.name for f in fields if not f.nullable and f.name not in df.columns]
    if missing:
        raise RawFileError(
            f"{label} file is missing required column(s): {', '.join(missing)}"
        )


def read_adsb_file(path: str | Path) -> pd.DataFrame:
    """Read a delivered ADS-B timeseries file (CSV/Parquet) and validate columns.

    Normalizes the displacement column: a file delivered with the corrected
    spelling (:data:`CORRECTED_DISPLACEMENT_NAME`) is renamed to the schema's
    verbatim typo name (:data:`~tdz.io.raw_schema.DISPLACEMENT_TYPO_ALIAS`) so
    downstream code deals with exactly one name. Column dtypes are coerced to
    the raw schema; unknown extra columns are preserved untouched.
    """
    df = _load_dataframe(path)
    if (CORRECTED_DISPLACEMENT_NAME in df.columns
            and DISPLACEMENT_TYPO_ALIAS not in df.columns):
        df = df.rename(columns={CORRECTED_DISPLACEMENT_NAME: DISPLACEMENT_TYPO_ALIAS})
    _require_columns(df, ADSB_FIELDS, label="ADS-B")
    return _apply_schema_dtypes(df, ADSB_FIELDS)


def read_qar_file(path: str | Path) -> list[QARRecord]:
    """Read a delivered QAR truth file (CSV/Parquet) into raw :class:`QARRecord` rows."""
    df = _load_dataframe(path)
    _require_columns(df, QAR_FIELDS, label="QAR")
    df = _apply_schema_dtypes(df, QAR_FIELDS)
    return qar_records_from_dataframe(df)


def qar_records_from_dataframe(df: pd.DataFrame) -> list[QARRecord]:
    """Build raw :class:`QARRecord` rows from a (schema-typed) QAR DataFrame."""
    records: list[QARRecord] = []
    for row in df.to_dict(orient="records"):
        records.append(
            QARRecord(
                registration=_opt_str(row.get("registration")) or "",
                landing_time=_opt_float(row.get("landing_time")) or float("nan"),
                destination_actual_icao=_opt_str(row.get("destination_actual_icao")) or "",
                destination_actual_runway=_opt_str(row.get("destination_actual_runway")) or "",
                longitude_at_touchdown=_opt_float(row.get("longitude_at_touchdown")) or float("nan"),
                latitude_at_touchdown=_opt_float(row.get("latitude_at_touchdown")) or float("nan"),
                aircraft_type=_opt_str(row.get("aircraft_type")) or "",
                destination_runway_length=_opt_float(row.get("destination_runway_length")),
                destination_runway_threshold_latitude=_opt_float(
                    row.get("destination_runway_threshold_latitude")),
                destination_runway_threshold_longitude=_opt_float(
                    row.get("destination_runway_threshold_longitude")),
                deceleration_at_60kt=_opt_float(row.get("deceleration_at_60kt")),
                cas_touchdown=_opt_float(row.get("cas_touchdown")),
                gs_touchdown=_opt_float(row.get("gs_touchdown")),
                aircraft_subtype=_opt_str(row.get("aircraft_subtype")),
                runway_length=_opt_float(row.get("runway_length")),
                HEXID=_opt_str(row.get("HEXID")),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Runway supplement table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunwaySupplementEntry:
    """Per-(airport, runway) reference geometry absent from the delivered files.

    ``elevation_m`` is meters in the datum named by ``elevation_datum``
    ("MSL" or "HAE"); ``geoid_undulation_m`` may be NaN (the datum resolver
    falls back to the EGM2008 grid lookup). ``threshold_lat``/``threshold_lon``
    and ``length_m`` are optional fallbacks used only when the delivered file's
    per-flight values are missing.
    """

    airport: str
    runway: str
    heading_deg: float
    elevation_m: float
    elevation_datum: str
    width_m: float
    geoid_undulation_m: float = float("nan")
    threshold_lat: Optional[float] = None
    threshold_lon: Optional[float] = None
    length_m: Optional[float] = None


def _supplement_key(airport: object, runway: object) -> tuple[str, str]:
    """Normalized (airport, runway) lookup key: upper-cased, stripped."""
    return (str(airport).strip().upper(), str(runway).strip().upper())


def read_runway_supplement(path: str | Path) -> dict[tuple[str, str], RunwaySupplementEntry]:
    """Read the runway supplement table (CSV/Parquet) into a lookup dict.

    Required columns: ``airport``, ``runway``, ``heading_deg``, ``elevation_m``,
    ``elevation_datum``, ``width_m``. Optional: ``geoid_undulation_m``,
    ``threshold_lat``, ``threshold_lon``, ``length_m``.
    """
    df = _load_dataframe(path)
    required = ("airport", "runway", "heading_deg", "elevation_m",
                "elevation_datum", "width_m")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RawFileError(
            f"runway supplement is missing required column(s): {', '.join(missing)}"
        )
    table: dict[tuple[str, str], RunwaySupplementEntry] = {}
    for row in df.to_dict(orient="records"):
        key = _supplement_key(row["airport"], row["runway"])
        table[key] = RunwaySupplementEntry(
            airport=key[0],
            runway=key[1],
            heading_deg=float(row["heading_deg"]),
            elevation_m=float(row["elevation_m"]),
            elevation_datum=str(row["elevation_datum"]).strip().upper(),
            width_m=float(row["width_m"]),
            geoid_undulation_m=_opt_float(row.get("geoid_undulation_m"))
            if _opt_float(row.get("geoid_undulation_m")) is not None else float("nan"),
            threshold_lat=_opt_float(row.get("threshold_lat")),
            threshold_lon=_opt_float(row.get("threshold_lon")),
            length_m=_opt_float(row.get("length_m")),
        )
    return table


# ---------------------------------------------------------------------------
# ADS-B rows -> FlightRecords
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SkippedFlight:
    """A flight the reader could not turn into a valid FlightRecord."""

    flight_id: str
    reason: str          # machine-readable code, see flight_records_from_adsb
    detail: str = ""


@dataclass(frozen=True)
class AdsbFlightMeta:
    """Per-flight join/stratification metadata carried alongside a FlightRecord.

    ``FlightRecord`` deliberately omits airframe identity; the QAR join
    (:func:`match_qar_to_flights`) and the tail-grouped validation split need
    it, so the reader surfaces it here.
    """

    flight_id: str
    hexid: str
    registration: str
    airport: str
    runway: str
    aircraft_type: str
    landing_timestamp: Optional[float]   # provider-derived; not ground truth
    n_rows: int


@dataclass(frozen=True)
class AdsbReadResult:
    """Everything produced by reading one delivered ADS-B file."""

    flights: list[FlightRecord]
    metas: dict[str, AdsbFlightMeta]     # flight_id -> meta
    skipped: list[SkippedFlight]
    units: RawUnits


#: Per-flight-constant columns whose within-flight consistency is validated.
_CONSTANT_COLUMNS = tuple(
    f.name for f in ADSB_FIELDS
    if f.constancy == "per_flight" and f.name != "flight_id"
)


def _first_valid(series: pd.Series) -> object:
    """First non-null value of a series, or None."""
    idx = series.first_valid_index()
    return None if idx is None else series.loc[idx]


def _opt_float(value: object) -> Optional[float]:
    """Optional float with pandas-NA/NaN/None -> None."""
    if value is None or value is pd.NA:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _opt_str(value: object) -> Optional[str]:
    """Optional string with pandas-NA/NaN/None -> None."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _inconsistent_constants(group: pd.DataFrame) -> list[str]:
    """Names of per-flight-constant columns with >1 distinct non-null value."""
    bad = []
    for name in _CONSTANT_COLUMNS:
        if name in group.columns and group[name].dropna().nunique() > 1:
            bad.append(name)
    return bad


def _build_runway(
    group: pd.DataFrame,
    supplement: Mapping[tuple[str, str], RunwaySupplementEntry],
    units: RawUnits,
) -> tuple[Optional[RunwayReference], Optional[SkippedFlight]]:
    """Assemble the RunwayReference for one flight group (or a skip reason)."""
    flight_id = str(group["flight_id"].iloc[0])
    airport = _opt_str(_first_valid(group["airport"]))
    runway_id = _opt_str(_first_valid(group["runway"]))
    if airport is None or runway_id is None:
        return None, SkippedFlight(flight_id, "missing_airport_or_runway")

    entry = supplement.get(_supplement_key(airport, runway_id))
    if entry is None:
        return None, SkippedFlight(
            flight_id, "missing_runway_supplement",
            f"no supplement entry for ({airport}, {runway_id})",
        )

    thr_lat = _opt_float(_first_valid(group["threshold_latitude"])) \
        if "threshold_latitude" in group.columns else None
    thr_lon = _opt_float(_first_valid(group["threshold_longitude"])) \
        if "threshold_longitude" in group.columns else None
    if thr_lat is None:
        thr_lat = entry.threshold_lat
    if thr_lon is None:
        thr_lon = entry.threshold_lon
    if thr_lat is None or thr_lon is None:
        return None, SkippedFlight(flight_id, "missing_threshold_coordinates")

    length_raw = _opt_float(_first_valid(group["runway_length"])) \
        if "runway_length" in group.columns else None
    if length_raw is not None:
        length_m = RawUnits.to_meters(length_raw, units.runway_length)
    elif entry.length_m is not None:
        length_m = entry.length_m
    else:
        return None, SkippedFlight(flight_id, "missing_runway_length")

    displacement = None
    if DISPLACEMENT_TYPO_ALIAS in group.columns:
        displacement = _opt_float(_first_valid(group[DISPLACEMENT_TYPO_ALIAS]))
    if displacement is not None:
        displacement = RawUnits.to_meters(displacement, units.displacement_length)
    displaced = bool(displacement is not None and displacement > 0.0)

    runway = RunwayReference(
        threshold_lat=thr_lat,
        threshold_lon=thr_lon,
        heading_deg=entry.heading_deg,
        elevation_m=entry.elevation_m,
        elevation_datum=entry.elevation_datum,
        geoid_undulation_m=entry.geoid_undulation_m,
        length_m=length_m,
        width_m=entry.width_m,
        displaced=displaced,
    )
    try:
        validate_runway_reference(runway)
    except InvalidRunwayReferenceError as exc:
        return None, SkippedFlight(flight_id, "invalid_runway_reference", str(exc))
    return runway, None


def _stream_times(group: pd.DataFrame, column: str) -> pd.Series:
    """Per-row stream times: ``column`` where finite, else the row timestamp.

    Async sources populate ``time_position``/``time_velocity``; co-timed
    sources leave them null and the shared ``timestamp`` is the sample time.
    Mixing is handled per row so a file with partial nulls still ingests.
    """
    base = pd.to_numeric(group["timestamp"], errors="coerce")
    if column in group.columns:
        specific = pd.to_numeric(group[column], errors="coerce")
        return specific.where(specific.notna(), base)
    return base


def _flight_record_from_group(
    group: pd.DataFrame,
    runway: RunwayReference,
    units: RawUnits,
    ads_b_source: str,
) -> tuple[Optional[FlightRecord], Optional[SkippedFlight]]:
    """Build the FlightRecord for one flight's rows (or a skip reason)."""
    flight_id = str(group["flight_id"].iloc[0])
    aircraft_type = _opt_str(_first_valid(group["model_icao"])) or ""

    # --- position stream ---------------------------------------------------
    pos_times = _stream_times(group, "time_position")
    lat = pd.to_numeric(group.get("latitude"), errors="coerce")
    lon = pd.to_numeric(group.get("longitude"), errors="coerce")
    pos_mask = pos_times.notna() & lat.notna() & lon.notna()
    if not bool(pos_mask.any()):
        return None, SkippedFlight(flight_id, "no_valid_position_samples")

    geo = pd.to_numeric(group.get("geometric_height"), errors="coerce")
    baro = pd.to_numeric(group.get("barometric_altitude"), errors="coerce")
    on_ground = group["on_ground"] if "on_ground" in group.columns else None

    position_times = pos_times[pos_mask].to_numpy(dtype=float)
    latitudes = lat[pos_mask].to_numpy(dtype=float)
    longitudes = lon[pos_mask].to_numpy(dtype=float)
    geometric = np.array(
        [RawUnits.to_meters(v, units.geometric_height) if not math.isnan(v) else v
         for v in geo[pos_mask].to_numpy(dtype=float)]
        if geo is not None else np.full(int(pos_mask.sum()), np.nan),
        dtype=float,
    )
    barometric = np.array(
        [RawUnits.to_meters(v, units.barometric_altitude) if not math.isnan(v) else v
         for v in baro[pos_mask].to_numpy(dtype=float)]
        if baro is not None else np.full(int(pos_mask.sum()), np.nan),
        dtype=float,
    )
    if on_ground is not None:
        flags = on_ground[pos_mask].fillna(False).to_numpy(dtype=bool)
    else:
        flags = np.zeros(position_times.shape, dtype=bool)

    (position_times, latitudes, longitudes, geometric, barometric, flags) = _sort_by_time(
        position_times, latitudes, longitudes, geometric, barometric, flags
    )

    # --- velocity stream (coalesced groundspeed) ----------------------------
    vel_times = _stream_times(group, "time_velocity")
    gs = pd.to_numeric(group.get("ground_speed"), errors="coerce")
    for fallback in ("ground_speed_airborne", "ground_speed_surface"):
        if fallback in group.columns:
            gs = gs.where(gs.notna(), pd.to_numeric(group[fallback], errors="coerce"))
    vel_mask = vel_times.notna() & gs.notna()
    if not bool(vel_mask.any()):
        return None, SkippedFlight(flight_id, "no_valid_velocity_samples")

    track = pd.to_numeric(group.get("track"), errors="coerce")
    vr = pd.to_numeric(group.get("barometric_vertical_rate"), errors="coerce")

    velocity_times = vel_times[vel_mask].to_numpy(dtype=float)
    groundspeeds = gs[vel_mask].to_numpy(dtype=float)          # stays knots
    tracks = (track[vel_mask].to_numpy(dtype=float)
              if track is not None else np.full(velocity_times.shape, np.nan))
    vertical_rates = (vr[vel_mask].to_numpy(dtype=float)       # stays ft/min
                      if vr is not None else np.full(velocity_times.shape, np.nan))

    (velocity_times, groundspeeds, tracks, vertical_rates) = _sort_by_time(
        velocity_times, groundspeeds, tracks, vertical_rates
    )

    record = FlightRecord(
        flight_id=flight_id,
        aircraft_type=aircraft_type,
        ads_b_source=ads_b_source,
        position_times=position_times,
        velocity_times=velocity_times,
        latitudes=latitudes,
        longitudes=longitudes,
        geometric_altitudes=geometric,
        barometric_altitudes=barometric,
        groundspeeds=groundspeeds,
        tracks=tracks,
        baro_vertical_rates=vertical_rates,
        on_ground_flags=flags,
        on_ground_transition_time=_on_ground_transition_time(position_times, flags),
        runway=runway,
    )
    return record, None


def flight_records_from_adsb(
    df: pd.DataFrame,
    *,
    supplement: Mapping[tuple[str, str], RunwaySupplementEntry],
    units: Optional[RawUnits] = None,
    ads_b_source: str = "aireon",
) -> AdsbReadResult:
    """Group ADS-B rows by flight and build FlightRecords + metadata.

    Skip reason codes: ``inconsistent_per_flight_constant``,
    ``missing_airport_or_runway``, ``missing_runway_supplement``,
    ``missing_threshold_coordinates``, ``missing_runway_length``,
    ``invalid_runway_reference``, ``no_valid_position_samples``,
    ``no_valid_velocity_samples``.

    Parameters
    ----------
    df:
        Schema-typed ADS-B rows (from :func:`read_adsb_file`).
    supplement:
        (airport, runway) -> :class:`RunwaySupplementEntry` lookup for the
        geometry the delivered file lacks (heading/elevation/width).
    units:
        Unit assumptions for the 'TODO: confirm' columns; defaults to the
        raw-schema assumptions (:class:`RawUnits`).
    ads_b_source:
        Source label stamped on every record (Req 8.4).
    """
    units = units or RawUnits()
    flights: list[FlightRecord] = []
    metas: dict[str, AdsbFlightMeta] = {}
    skipped: list[SkippedFlight] = []

    for flight_id, group in df.groupby("flight_id", sort=True):
        fid = str(flight_id)
        bad = _inconsistent_constants(group)
        if bad:
            skipped.append(SkippedFlight(
                fid, "inconsistent_per_flight_constant", ", ".join(bad)))
            continue

        runway, skip = _build_runway(group, supplement, units)
        if skip is not None:
            skipped.append(skip)
            continue
        assert runway is not None

        record, skip = _flight_record_from_group(group, runway, units, ads_b_source)
        if skip is not None:
            skipped.append(skip)
            continue
        assert record is not None

        flights.append(record)
        metas[fid] = AdsbFlightMeta(
            flight_id=fid,
            hexid=(_opt_str(_first_valid(group["hexid"])) or "").upper(),
            registration=_opt_str(_first_valid(group["aircraft_registration"])) or "",
            airport=_opt_str(_first_valid(group["airport"])) or "",
            runway=_opt_str(_first_valid(group["runway"])) or "",
            aircraft_type=record.aircraft_type,
            landing_timestamp=_opt_float(_first_valid(group["landing_timestamp"]))
            if "landing_timestamp" in group.columns else None,
            n_rows=int(len(group)),
        )

    return AdsbReadResult(flights=flights, metas=metas, skipped=skipped, units=units)


def read_adsb_flights(
    path: str | Path,
    *,
    supplement: Mapping[tuple[str, str], RunwaySupplementEntry],
    units: Optional[RawUnits] = None,
    ads_b_source: str = "aireon",
) -> AdsbReadResult:
    """One-call convenience: :func:`read_adsb_file` + :func:`flight_records_from_adsb`."""
    return flight_records_from_adsb(
        read_adsb_file(path),
        supplement=supplement,
        units=units,
        ads_b_source=ads_b_source,
    )


# ---------------------------------------------------------------------------
# QAR truth matching (file-level join; clock alignment is Task 21)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QARMatchResult:
    """Outcome of joining QAR truth rows to ADS-B flights."""

    truths: list[QARTruthRecord]
    unmatched_flights: list[str] = field(default_factory=list)   # no QAR candidate
    ambiguous_flights: list[str] = field(default_factory=list)   # >1 candidate, no tiebreak


def match_qar_to_flights(
    metas: Mapping[str, AdsbFlightMeta] | Sequence[AdsbFlightMeta],
    qar_records: Sequence[QARRecord],
    *,
    time_tolerance_s: float = 300.0,
) -> QARMatchResult:
    """Join QAR truth rows to ADS-B flights by airframe + landing-time proximity.

    Matching follows :data:`~tdz.io.raw_schema.JOIN_KEYS`: airframe (Mode S
    HEXID, case-insensitive) first, registration as the fallback. When several
    QAR rows match the airframe, the flight's provider ``landing_timestamp``
    breaks the tie (nearest ``landing_time`` within ``time_tolerance_s``); with
    no usable timestamp and >1 candidate the flight is reported ambiguous
    rather than guessed. The provider landing timestamp is used ONLY for
    candidate disambiguation, never as truth — and the tolerance is deliberately
    coarse because QAR-vs-ADS-B clock offsets are unresolved at read time
    (alignment is Task 21).

    Emitted records carry ``clock_offset_estimate=None`` and an empty
    ``clock_offset_quality``: the offset is estimated later by clock alignment.
    """
    meta_list = list(metas.values()) if isinstance(metas, Mapping) else list(metas)
    truths: list[QARTruthRecord] = []
    unmatched: list[str] = []
    ambiguous: list[str] = []

    by_hexid: dict[str, list[QARRecord]] = {}
    by_registration: dict[str, list[QARRecord]] = {}
    for rec in qar_records:
        if rec.HEXID:
            by_hexid.setdefault(rec.HEXID.strip().upper(), []).append(rec)
        if rec.registration:
            by_registration.setdefault(rec.registration.strip().upper(), []).append(rec)

    for meta in meta_list:
        candidates = by_hexid.get(meta.hexid.strip().upper(), []) if meta.hexid else []
        if not candidates and meta.registration:
            candidates = by_registration.get(meta.registration.strip().upper(), [])
        if not candidates:
            unmatched.append(meta.flight_id)
            continue

        chosen: Optional[QARRecord] = None
        if len(candidates) == 1:
            chosen = candidates[0]
            if (meta.landing_timestamp is not None
                    and math.isfinite(chosen.landing_time)
                    and abs(chosen.landing_time - meta.landing_timestamp) > time_tolerance_s):
                chosen = None  # single candidate but wrong landing
        elif meta.landing_timestamp is not None:
            in_window = [
                r for r in candidates
                if math.isfinite(r.landing_time)
                and abs(r.landing_time - meta.landing_timestamp) <= time_tolerance_s
            ]
            if len(in_window) >= 1:
                chosen = min(
                    in_window,
                    key=lambda r: abs(r.landing_time - meta.landing_timestamp),
                )
        else:
            ambiguous.append(meta.flight_id)
            continue

        if chosen is None:
            unmatched.append(meta.flight_id)
            continue

        truths.append(QARTruthRecord(
            flight_id=meta.flight_id,
            touchdown_time_qar=float(chosen.landing_time),
            touchdown_lat=float(chosen.latitude_at_touchdown),
            touchdown_lon=float(chosen.longitude_at_touchdown),
            clock_offset_estimate=None,
            clock_offset_quality=_CLOCK_QUALITY_UNALIGNED,
            aircraft_type=chosen.aircraft_type,
            runway_id=chosen.destination_actual_runway,
            airport_id=chosen.destination_actual_icao,
            tail_number=chosen.registration,
        ))

    return QARMatchResult(
        truths=truths, unmatched_flights=unmatched, ambiguous_flights=ambiguous
    )
