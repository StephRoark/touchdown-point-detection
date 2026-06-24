"""Parse ADS-B source formats into the internal FlightRecord (Task 9.1).

Two source formats are parsed into the single pipeline-internal
:class:`~tdz.models.FlightRecord` contract so everything downstream is
source-agnostic (Req 8.1, 8.2):

* **Aireon (asynchronous)** -- a stream of :class:`~tdz.models.AireonMessage`
  position and velocity records sharing a ``flight_id`` but carrying SEPARATE
  position and velocity timestamps. :func:`parse_aireon_messages` keeps the
  distinct ``position_times`` and ``velocity_times`` arrays and never merges
  them (Req 8.3, 10.1; consistent with Task 8 / Property 4).

* **FlightRadar24 (co-timed)** -- :class:`~tdz.models.FR24Record` rows with a
  single timestamp per row. :func:`parse_fr24_records` sets
  ``position_times == velocity_times`` (co-timed) but still stores the values in
  *both* arrays so the downstream contract is uniform.

The ADS-B source string is recorded on the resulting ``FlightRecord``
(``ads_b_source``) so every estimate can report which source it came from
(Req 8.4). Runway geometry (:class:`~tdz.models.RunwayReference`) and the ICAO
aircraft type are accepted as inputs and joined onto the flight; the full
QAR-by-airframe join is validation (Task 21), but a light optional lookup
helper (:func:`find_qar_truth`) is provided using :data:`tdz.io.raw_schema.JOIN_KEYS`.

Vertical-datum / barometric-vs-geometric invariant
---------------------------------------------------
Barometric altitude is NEVER placed into the geometric-altitude array (Req 8.8).
For Aireon, geometric and barometric altitudes arrive in distinct fields. For
FR24 the single ``altitude_m`` is routed by ``altitude_kind``: only
``"geometric"`` populates ``geometric_altitudes``; ``"barometric"`` and
``"unknown"`` populate ``barometric_altitudes`` and leave the geometric array
``NaN`` (so a no-geometric source has an all-``NaN`` geometric array, which the
gating in :mod:`tdz.io.gating` and Property 20 rely on).

Units convention
----------------
The :class:`AireonMessage` / :class:`FR24Record` altitude fields are already in
meters (``*_m`` / ``altitude_m``), so no altitude conversion happens here.
Groundspeed stays in knots and track in degrees true on the ``FlightRecord``
(the same native units Task 8's interpolation consumes, converting with
``KNOTS_TO_MPS`` only where SI is needed). Barometric vertical rate stays in
ft/min and may be ``NaN`` (Req 9.1). Missing optional fields become ``NaN``
(or ``False`` for the boolean on-ground flag).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from tdz.io.raw_schema import JOIN_KEYS, QARRecord
from tdz.models import AireonMessage, FlightRecord, FR24Record, RunwayReference

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = [
    "parse_aireon_messages",
    "parse_fr24_records",
    "find_qar_truth",
    "aireon_messages_from_dataframe",
    "fr24_records_from_dataframe",
]


def _f(value: Optional[float]) -> float:
    """Coerce an optional float to ``float`` with ``None`` -> ``NaN``."""
    if value is None:
        return float("nan")
    return float(value)


def _on_ground_transition_time(
    times: np.ndarray, on_ground: np.ndarray
) -> Optional[float]:
    """First time at which the on-ground flag transitions False -> True.

    Returns the timestamp of the first ``True`` sample that is preceded by a
    ``False`` sample (a genuine air->ground transition). If the very first
    sample is already on-ground (no preceding airborne sample) or there is no
    transition, returns ``None``. The on-ground flag is only ever an upper
    time-bound for the coarse bracket (design); this helper just surfaces it.
    """
    if times.size == 0:
        return None
    flags = on_ground.astype(bool)
    for i in range(1, flags.size):
        if flags[i] and not flags[i - 1]:
            return float(times[i])
    return None


def _sort_by_time(times: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Stable-sort ``times`` and reorder the parallel ``arrays`` to match."""
    order = np.argsort(times, kind="stable")
    return (times[order], *(arr[order] for arr in arrays))


def parse_aireon_messages(
    messages: Sequence[AireonMessage],
    *,
    runway: RunwayReference,
    aircraft_type: str,
    ads_b_source: str = "aireon",
    flight_id: Optional[str] = None,
) -> FlightRecord:
    """Parse an Aireon async message stream into a :class:`FlightRecord`.

    Position messages (``message_type == "position"``) and velocity messages
    (``message_type == "velocity"``) are separated into their own time-ordered
    arrays; the two timebases are preserved distinctly and never merged
    (Req 8.3, 10.1). Each stream is stable-sorted by timestamp.

    Parameters
    ----------
    messages:
        The flight's Aireon messages (mixed position/velocity), all sharing one
        ``flight_id``.
    runway:
        Runway geometry joined onto the flight.
    aircraft_type:
        ICAO type designator joined onto the flight.
    ads_b_source:
        Source label recorded on the record (Req 8.4); defaults to ``"aireon"``.
    flight_id:
        Optional explicit flight id; when omitted it is taken from the messages.

    Returns
    -------
    FlightRecord
        With separate ``position_times`` / ``velocity_times`` arrays.

    Raises
    ------
    ValueError
        When ``messages`` is empty, a ``message_type`` is unknown, or no
        ``flight_id`` can be determined.
    """
    if len(messages) == 0:
        raise ValueError("cannot parse an empty Aireon message stream")

    resolved_flight_id = flight_id
    if resolved_flight_id is None:
        ids = {m.flight_id for m in messages}
        if len(ids) != 1:
            raise ValueError(
                f"expected a single flight_id in the message stream, got {sorted(ids)!r}"
            )
        resolved_flight_id = next(iter(ids))

    pos = [m for m in messages if m.message_type == "position"]
    vel = [m for m in messages if m.message_type == "velocity"]
    unknown = {
        m.message_type
        for m in messages
        if m.message_type not in ("position", "velocity")
    }
    if unknown:
        raise ValueError(f"unknown Aireon message_type(s): {sorted(unknown)!r}")

    # Position stream.
    position_times = np.array([m.timestamp for m in pos], dtype=float)
    latitudes = np.array([_f(m.latitude) for m in pos], dtype=float)
    longitudes = np.array([_f(m.longitude) for m in pos], dtype=float)
    geometric_altitudes = np.array(
        [_f(m.geometric_altitude_m) for m in pos], dtype=float
    )
    barometric_altitudes = np.array(
        [_f(m.barometric_altitude_m) for m in pos], dtype=float
    )
    on_ground_flags = np.array(
        [bool(m.on_ground) if m.on_ground is not None else False for m in pos],
        dtype=bool,
    )
    (
        position_times,
        latitudes,
        longitudes,
        geometric_altitudes,
        barometric_altitudes,
        on_ground_flags,
    ) = _sort_by_time(
        position_times,
        latitudes,
        longitudes,
        geometric_altitudes,
        barometric_altitudes,
        on_ground_flags,
    )

    # Velocity stream (distinct timebase).
    velocity_times = np.array([m.timestamp for m in vel], dtype=float)
    groundspeeds = np.array([_f(m.groundspeed_kt) for m in vel], dtype=float)
    tracks = np.array([_f(m.track_deg) for m in vel], dtype=float)
    baro_vertical_rates = np.array(
        [_f(m.baro_vertical_rate_ftmin) for m in vel], dtype=float
    )
    velocity_times, groundspeeds, tracks, baro_vertical_rates = _sort_by_time(
        velocity_times, groundspeeds, tracks, baro_vertical_rates
    )

    return FlightRecord(
        flight_id=resolved_flight_id,
        aircraft_type=aircraft_type,
        ads_b_source=ads_b_source,
        position_times=position_times,
        velocity_times=velocity_times,
        latitudes=latitudes,
        longitudes=longitudes,
        geometric_altitudes=geometric_altitudes,
        barometric_altitudes=barometric_altitudes,
        groundspeeds=groundspeeds,
        tracks=tracks,
        baro_vertical_rates=baro_vertical_rates,
        on_ground_flags=on_ground_flags,
        on_ground_transition_time=_on_ground_transition_time(
            position_times, on_ground_flags
        ),
        runway=runway,
    )


def parse_fr24_records(
    records: Sequence[FR24Record],
    *,
    runway: RunwayReference,
    aircraft_type: str,
    ads_b_source: str = "flightradar24",
    flight_id: Optional[str] = None,
) -> FlightRecord:
    """Parse co-timed FlightRadar24 rows into a :class:`FlightRecord`.

    FR24 rows are co-timed: each row carries one timestamp for position and
    velocity. The single timestamp populates BOTH ``position_times`` and
    ``velocity_times`` (as separate array objects) so the downstream contract is
    uniform with the async case, while genuinely being co-timed
    (``position_times == velocity_times``).

    Altitude routing (Req 8.8): the single ``altitude_m`` is placed into
    ``geometric_altitudes`` only when ``altitude_kind == "geometric"``; for
    ``"barometric"`` or ``"unknown"`` it goes into ``barometric_altitudes`` and
    the geometric array is left ``NaN`` -- barometric altitude is never
    substituted into the geometric field.

    Parameters
    ----------
    records:
        The flight's FR24 rows.
    runway:
        Runway geometry joined onto the flight.
    aircraft_type:
        ICAO type designator joined onto the flight.
    ads_b_source:
        Source label recorded on the record (Req 8.4); defaults to
        ``"flightradar24"``.
    flight_id:
        Optional explicit flight id; when omitted it is taken from the rows.

    Returns
    -------
    FlightRecord
        With ``position_times == velocity_times`` (co-timed).

    Raises
    ------
    ValueError
        When ``records`` is empty or no single ``flight_id`` can be determined.
    """
    if len(records) == 0:
        raise ValueError("cannot parse an empty FR24 record stream")

    resolved_flight_id = flight_id
    if resolved_flight_id is None:
        ids = {r.flight_id for r in records}
        if len(ids) != 1:
            raise ValueError(
                f"expected a single flight_id in the FR24 rows, got {sorted(ids)!r}"
            )
        resolved_flight_id = next(iter(ids))

    times = np.array([r.timestamp for r in records], dtype=float)
    latitudes = np.array([_f(r.latitude) for r in records], dtype=float)
    longitudes = np.array([_f(r.longitude) for r in records], dtype=float)
    groundspeeds = np.array([_f(r.groundspeed_kt) for r in records], dtype=float)
    tracks = np.array([_f(r.track_deg) for r in records], dtype=float)
    baro_vertical_rates = np.array(
        [_f(r.vertical_rate_ftmin) for r in records], dtype=float
    )
    on_ground_flags = np.array([bool(r.on_ground) for r in records], dtype=bool)

    # Route altitude by kind; barometric is never put into the geometric array.
    geometric_altitudes = np.full(times.shape, np.nan, dtype=float)
    barometric_altitudes = np.full(times.shape, np.nan, dtype=float)
    for i, r in enumerate(records):
        if r.altitude_kind == "geometric":
            geometric_altitudes[i] = _f(r.altitude_m)
        else:  # "barometric" | "unknown" -> never the geometric field (Req 8.8)
            barometric_altitudes[i] = _f(r.altitude_m)

    (
        times,
        latitudes,
        longitudes,
        geometric_altitudes,
        barometric_altitudes,
        groundspeeds,
        tracks,
        baro_vertical_rates,
        on_ground_flags,
    ) = _sort_by_time(
        times,
        latitudes,
        longitudes,
        geometric_altitudes,
        barometric_altitudes,
        groundspeeds,
        tracks,
        baro_vertical_rates,
        on_ground_flags,
    )

    return FlightRecord(
        flight_id=resolved_flight_id,
        aircraft_type=aircraft_type,
        ads_b_source=ads_b_source,
        position_times=times.copy(),
        velocity_times=times.copy(),  # co-timed: same values, separate arrays
        latitudes=latitudes,
        longitudes=longitudes,
        geometric_altitudes=geometric_altitudes,
        barometric_altitudes=barometric_altitudes,
        groundspeeds=groundspeeds,
        tracks=tracks,
        baro_vertical_rates=baro_vertical_rates,
        on_ground_flags=on_ground_flags,
        on_ground_transition_time=_on_ground_transition_time(times, on_ground_flags),
        runway=runway,
    )


def find_qar_truth(
    qar_records: Sequence[QARRecord],
    *,
    hexid: Optional[str] = None,
    registration: Optional[str] = None,
) -> Optional[QARRecord]:
    """Light optional QAR lookup by airframe (Mode S HEXID) or registration.

    The authoritative QAR join (and clock alignment) is the validation harness
    (Task 21); this helper is a convenience for tests and ingest-time metadata
    enrichment, driven by :data:`tdz.io.raw_schema.JOIN_KEYS`. It matches on the
    ``airframe`` key (``HEXID``) first, then falls back to the ``registration``
    key. Returns the first matching record, or ``None`` if there is no match.

    Parameters
    ----------
    qar_records:
        Candidate QAR rows.
    hexid:
        24-bit Mode S address to match against ``QARRecord.HEXID``.
    registration:
        Tail number to match against ``QARRecord.registration``.

    Raises
    ------
    ValueError
        When neither ``hexid`` nor ``registration`` is supplied.
    """
    if hexid is None and registration is None:
        raise ValueError("supply at least one of hexid or registration to join on")

    # JOIN_KEYS["airframe"] == ("hexid", "HEXID"); ["registration"] == (..., "registration").
    _, qar_hexid_field = JOIN_KEYS["airframe"]
    _, qar_reg_field = JOIN_KEYS["registration"]

    if hexid is not None:
        for rec in qar_records:
            value = getattr(rec, qar_hexid_field, None)
            if value is not None and value == hexid:
                return rec

    if registration is not None:
        for rec in qar_records:
            value = getattr(rec, qar_reg_field, None)
            if value is not None and value == registration:
                return rec

    return None


# ---------------------------------------------------------------------------
# Optional pandas adapters (lazy import; numpy-only core does not need pandas)
# ---------------------------------------------------------------------------


def aireon_messages_from_dataframe(df: "pd.DataFrame") -> list[AireonMessage]:
    """Build :class:`AireonMessage` rows from a DataFrame (optional, lazy pandas).

    Columns mirror the :class:`AireonMessage` fields (``flight_id``,
    ``message_type``, ``timestamp`` and the optional position/velocity fields).
    Missing optional columns are treated as ``None``. pandas is imported lazily
    so the numpy-only core never requires it; install pandas to use this
    adapter. The returned list can be fed to :func:`parse_aireon_messages`.
    """
    fields = (
        "flight_id",
        "message_type",
        "timestamp",
        "latitude",
        "longitude",
        "geometric_altitude_m",
        "barometric_altitude_m",
        "on_ground",
        "groundspeed_kt",
        "track_deg",
        "baro_vertical_rate_ftmin",
    )
    out: list[AireonMessage] = []
    for row in df.to_dict(orient="records"):
        kwargs = {k: row[k] for k in fields if k in row and _notnull(row[k])}
        out.append(AireonMessage(**kwargs))
    return out


def fr24_records_from_dataframe(df: "pd.DataFrame") -> list[FR24Record]:
    """Build :class:`FR24Record` rows from a DataFrame (optional, lazy pandas).

    Columns mirror the :class:`FR24Record` fields. Missing optional columns are
    treated as ``None``. pandas is imported lazily so the numpy-only core never
    requires it. The returned list can be fed to :func:`parse_fr24_records`.
    """
    required = (
        "flight_id",
        "timestamp",
        "latitude",
        "longitude",
        "altitude_m",
        "altitude_kind",
        "groundspeed_kt",
        "track_deg",
        "on_ground",
    )
    optional = ("vertical_rate_ftmin",)
    out: list[FR24Record] = []
    for row in df.to_dict(orient="records"):
        kwargs = {k: row[k] for k in required}
        for k in optional:
            if k in row and _notnull(row[k]):
                kwargs[k] = row[k]
        out.append(FR24Record(**kwargs))
    return out


def _notnull(value: object) -> bool:
    """Return whether a (possibly pandas/NaN) value is present."""
    if value is None:
        return False
    try:
        return not (isinstance(value, float) and np.isnan(value))
    except (TypeError, ValueError):
        return True
