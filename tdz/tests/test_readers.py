"""Tests for the delivered-file readers (tdz.io.readers).

Covers: CSV/Parquet loading, required-column validation, the displacement
typo-alias normalization, per-flight grouping with constancy checks, unit
conversion at the raw->SI boundary, runway-supplement joining with explicit
skip reasons, async-vs-co-timed stream handling, and the QAR-by-airframe
match with landing-time disambiguation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tdz.io.raw_schema import DISPLACEMENT_TYPO_ALIAS, QARRecord
from tdz.io.readers import (
    CORRECTED_DISPLACEMENT_NAME,
    FT_TO_M,
    AdsbFlightMeta,
    RawFileError,
    RawUnits,
    RunwaySupplementEntry,
    flight_records_from_adsb,
    match_qar_to_flights,
    read_adsb_file,
    read_adsb_flights,
    read_qar_file,
    read_runway_supplement,
)

# KJFK 04L-ish geometry (threshold coordinates only need to be plausible).
THR_LAT, THR_LON = 40.622021, -73.785584
HEADING = 43.9


def _supplement() -> dict:
    return {
        ("KJFK", "04L"): RunwaySupplementEntry(
            airport="KJFK",
            runway="04L",
            heading_deg=HEADING,
            elevation_m=3.4,
            elevation_datum="MSL",
            width_m=61.0,
            geoid_undulation_m=-32.8,
        ),
    }


def _adsb_rows(
    flight_id: str = "F1",
    *,
    n: int = 6,
    async_times: bool = True,
    runway_length: float = 3682.0,
    displacement_col: str = DISPLACEMENT_TYPO_ALIAS,
    airport: str = "KJFK",
    runway: str = "04L",
) -> list[dict]:
    """Rows mimicking the delivered ADS-B layout for one flight."""
    rows = []
    for i in range(n):
        t = 1000.0 + 4.5 * i
        rows.append({
            "flight_id": flight_id,
            "timestamp": t,
            "hexid": "A1B2C3",
            "time_position": t + 0.3 if async_times else np.nan,
            "time_velocity": t + 1.1 if async_times else np.nan,
            "latitude": THR_LAT - 0.001 * (n - i),
            "longitude": THR_LON - 0.001 * (n - i),
            "ground_speed": 140.0 - 2.0 * i,
            "airport": airport,
            "runway": runway,
            "on_ground": i >= n - 2,
            "landing_timestamp": 1000.0 + 4.5 * (n - 2),
            "landing_latitude": THR_LAT,
            "landing_longitude": THR_LON,
            "aircraft_registration": "N12345",
            "runway_length": runway_length,
            "model_icao": "B738",
            "aircraft_model_number_and_subtype": "737-800",
            "threshold_latitude": THR_LAT,
            "threshold_longitude": THR_LON,
            displacement_col: 0.0,
            "opposite_displacement_length": 0.0,
            "lda_day": 3682.0,
            "ground_speed_airborne": np.nan,
            "ground_speed_surface": np.nan,
            "barometric_altitude": 200.0 - 30.0 * i,   # feet (assumed)
            "barometric_vertical_rate": -700.0 + 100.0 * i,
            "geometric_height": 250.0 - 40.0 * i,      # feet HAE (assumed)
            "selected_altitude": 3000.0,
            "track": 44.0,
        })
    return rows


def _qar_row(**overrides) -> dict:
    row = {
        "registration": "N12345",
        "landing_time": 1018.0,
        "destination_actual_icao": "KJFK",
        "destination_actual_runway": "04L",
        "destination_runway_length": 3682.0,
        "longitude_at_touchdown": THR_LON + 0.004,
        "latitude_at_touchdown": THR_LAT + 0.004,
        "destination_runway_threshold_latitude": THR_LAT,
        "destination_runway_threshold_longitude": THR_LON,
        "deceleration_at_60kt": -2.1,
        "cas_touchdown": 138.0,
        "gs_touchdown": 142.0,
        "aircraft_type": "B738",
        "aircraft_subtype": "737-800",
        "runway_length": 3682.0,
        "HEXID": "A1B2C3",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# File loading + schema validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_adsb_csv_roundtrip(tmp_path):
    path = tmp_path / "adsb.csv"
    pd.DataFrame(_adsb_rows()).to_csv(path, index=False)
    df = read_adsb_file(path)
    assert len(df) == 6
    assert df["ground_speed"].dtype == np.float64
    assert str(df["on_ground"].dtype) == "boolean"
    assert bool(df["on_ground"].iloc[-1])


@pytest.mark.unit
def test_read_adsb_missing_required_column_raises(tmp_path):
    rows = _adsb_rows()
    for r in rows:
        del r["hexid"]
    path = tmp_path / "adsb.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(RawFileError, match="hexid"):
        read_adsb_file(path)


@pytest.mark.unit
def test_read_adsb_unknown_extension_raises(tmp_path):
    path = tmp_path / "adsb.xlsx"
    path.write_text("not really")
    with pytest.raises(RawFileError, match="unsupported"):
        read_adsb_file(path)


@pytest.mark.unit
def test_read_adsb_missing_file_raises(tmp_path):
    with pytest.raises(RawFileError, match="does not exist"):
        read_adsb_file(tmp_path / "nope.csv")


@pytest.mark.unit
def test_corrected_displacement_spelling_normalized(tmp_path):
    rows = _adsb_rows(displacement_col=CORRECTED_DISPLACEMENT_NAME)
    path = tmp_path / "adsb.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    df = read_adsb_file(path)
    assert DISPLACEMENT_TYPO_ALIAS in df.columns
    assert CORRECTED_DISPLACEMENT_NAME not in df.columns


@pytest.mark.unit
def test_read_adsb_parquet_roundtrip(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "adsb.parquet"
    pd.DataFrame(_adsb_rows()).to_parquet(path, index=False)
    df = read_adsb_file(path)
    assert len(df) == 6


# ---------------------------------------------------------------------------
# Flight assembly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flight_records_happy_path_async(tmp_path):
    path = tmp_path / "adsb.csv"
    pd.DataFrame(_adsb_rows() + _adsb_rows("F2")).to_csv(path, index=False)
    result = read_adsb_flights(path, supplement=_supplement())

    assert [f.flight_id for f in result.flights] == ["F1", "F2"]
    assert result.skipped == []
    flight = result.flights[0]
    # async: distinct position/velocity timebases from time_position/-velocity
    assert flight.position_times[0] == pytest.approx(1000.3)
    assert flight.velocity_times[0] == pytest.approx(1001.1)
    # geometric height converted ft -> m under the default assumption
    assert flight.geometric_altitudes[0] == pytest.approx(250.0 * FT_TO_M)
    # groundspeed stays in knots
    assert flight.groundspeeds[0] == pytest.approx(140.0)
    # runway joined from file + supplement
    assert flight.runway.heading_deg == pytest.approx(HEADING)
    assert flight.runway.length_m == pytest.approx(3682.0)
    assert flight.runway.width_m == pytest.approx(61.0)
    # on-ground transition surfaced (flag flips at sample n-2)
    assert flight.on_ground_transition_time == pytest.approx(1018.3)
    # metadata for the QAR join / tail split
    meta = result.metas["F1"]
    assert meta.hexid == "A1B2C3"
    assert meta.registration == "N12345"
    assert meta.landing_timestamp == pytest.approx(1018.0)


@pytest.mark.unit
def test_flight_records_co_timed_falls_back_to_timestamp():
    df = pd.DataFrame(_adsb_rows(async_times=False))
    result = flight_records_from_adsb(df, supplement=_supplement())
    flight = result.flights[0]
    np.testing.assert_allclose(flight.position_times, flight.velocity_times)
    assert flight.position_times[0] == pytest.approx(1000.0)


@pytest.mark.unit
def test_runway_length_feet_units_convert():
    df = pd.DataFrame(_adsb_rows(runway_length=12000.0))  # feet-valued data
    result = flight_records_from_adsb(
        df, supplement=_supplement(), units=RawUnits(runway_length="ft")
    )
    assert result.flights[0].runway.length_m == pytest.approx(12000.0 * FT_TO_M)


@pytest.mark.unit
def test_runway_length_feet_misread_as_meters_is_rejected():
    """A 12,000 ft runway read as meters exceeds the 6000 m bound -> skipped."""
    df = pd.DataFrame(_adsb_rows(runway_length=12000.0))
    result = flight_records_from_adsb(df, supplement=_supplement())  # assumes m
    assert result.flights == []
    assert result.skipped[0].reason == "invalid_runway_reference"


@pytest.mark.unit
def test_missing_supplement_entry_skips_with_reason():
    df = pd.DataFrame(_adsb_rows(airport="KLAX", runway="25L"))
    result = flight_records_from_adsb(df, supplement=_supplement())
    assert result.flights == []
    assert result.skipped[0].reason == "missing_runway_supplement"
    assert "KLAX" in result.skipped[0].detail


@pytest.mark.unit
def test_inconsistent_per_flight_constant_skips():
    rows = _adsb_rows()
    rows[2]["runway"] = "22R"  # constant column varies within the flight
    result = flight_records_from_adsb(pd.DataFrame(rows), supplement=_supplement())
    assert result.flights == []
    assert result.skipped[0].reason == "inconsistent_per_flight_constant"
    assert "runway" in result.skipped[0].detail


@pytest.mark.unit
def test_no_position_samples_skips():
    rows = _adsb_rows()
    for r in rows:
        r["latitude"] = np.nan
    result = flight_records_from_adsb(pd.DataFrame(rows), supplement=_supplement())
    assert result.skipped[0].reason == "no_valid_position_samples"


@pytest.mark.unit
def test_groundspeed_coalesces_airborne_and_surface_columns():
    rows = _adsb_rows()
    for i, r in enumerate(rows):
        r["ground_speed"] = np.nan
        if i < 3:
            r["ground_speed_airborne"] = 150.0 - i
        else:
            r["ground_speed_surface"] = 100.0 - i
    result = flight_records_from_adsb(pd.DataFrame(rows), supplement=_supplement())
    flight = result.flights[0]
    assert flight.groundspeeds.size == 6
    assert flight.groundspeeds[0] == pytest.approx(150.0)
    assert flight.groundspeeds[-1] == pytest.approx(95.0)


@pytest.mark.unit
def test_displaced_threshold_flag_set_when_displacement_positive():
    rows = _adsb_rows()
    for r in rows:
        r[DISPLACEMENT_TYPO_ALIAS] = 150.0
    result = flight_records_from_adsb(pd.DataFrame(rows), supplement=_supplement())
    assert result.flights[0].runway.displaced is True


# ---------------------------------------------------------------------------
# Runway supplement reading
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_runway_supplement_csv(tmp_path):
    path = tmp_path / "runways.csv"
    pd.DataFrame([{
        "airport": "kjfk", "runway": "04l", "heading_deg": HEADING,
        "elevation_m": 3.4, "elevation_datum": "msl", "width_m": 61.0,
        "geoid_undulation_m": -32.8,
    }]).to_csv(path, index=False)
    table = read_runway_supplement(path)
    entry = table[("KJFK", "04L")]  # keys and datum normalized to upper-case
    assert entry.elevation_datum == "MSL"
    assert entry.geoid_undulation_m == pytest.approx(-32.8)


@pytest.mark.unit
def test_read_runway_supplement_missing_column_raises(tmp_path):
    path = tmp_path / "runways.csv"
    pd.DataFrame([{"airport": "KJFK", "runway": "04L"}]).to_csv(path, index=False)
    with pytest.raises(RawFileError, match="heading_deg"):
        read_runway_supplement(path)


# ---------------------------------------------------------------------------
# QAR reading + matching
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_qar_csv(tmp_path):
    path = tmp_path / "qar.csv"
    pd.DataFrame([_qar_row()]).to_csv(path, index=False)
    records = read_qar_file(path)
    assert len(records) == 1
    assert isinstance(records[0], QARRecord)
    assert records[0].HEXID == "A1B2C3"
    assert records[0].gs_touchdown == pytest.approx(142.0)


@pytest.mark.unit
def test_read_qar_missing_required_column_raises(tmp_path):
    row = _qar_row()
    del row["landing_time"]
    path = tmp_path / "qar.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    with pytest.raises(RawFileError, match="landing_time"):
        read_qar_file(path)


def _meta(flight_id="F1", hexid="A1B2C3", registration="N12345",
          landing_timestamp=1018.0) -> AdsbFlightMeta:
    return AdsbFlightMeta(
        flight_id=flight_id, hexid=hexid, registration=registration,
        airport="KJFK", runway="04L", aircraft_type="B738",
        landing_timestamp=landing_timestamp, n_rows=6,
    )


@pytest.mark.unit
def test_qar_match_by_hexid():
    qar = [QARRecord(**_qar_kwargs())]
    result = match_qar_to_flights([_meta()], qar)
    assert len(result.truths) == 1
    truth = result.truths[0]
    assert truth.flight_id == "F1"
    assert truth.tail_number == "N12345"
    assert truth.clock_offset_estimate is None
    assert result.unmatched_flights == []


@pytest.mark.unit
def test_qar_match_registration_fallback():
    qar = [QARRecord(**_qar_kwargs(HEXID=None))]
    result = match_qar_to_flights([_meta(hexid="")], qar)
    assert len(result.truths) == 1


@pytest.mark.unit
def test_qar_match_time_disambiguates_multiple_landings():
    qar = [
        QARRecord(**_qar_kwargs(landing_time=1018.0)),
        QARRecord(**_qar_kwargs(landing_time=99000.0)),  # same tail, later leg
    ]
    result = match_qar_to_flights([_meta(landing_timestamp=1020.0)], qar)
    assert len(result.truths) == 1
    assert result.truths[0].touchdown_time_qar == pytest.approx(1018.0)


@pytest.mark.unit
def test_qar_match_ambiguous_without_timestamp():
    qar = [
        QARRecord(**_qar_kwargs(landing_time=1018.0)),
        QARRecord(**_qar_kwargs(landing_time=99000.0)),
    ]
    result = match_qar_to_flights([_meta(landing_timestamp=None)], qar)
    assert result.truths == []
    assert result.ambiguous_flights == ["F1"]


@pytest.mark.unit
def test_qar_match_outside_tolerance_unmatched():
    qar = [QARRecord(**_qar_kwargs(landing_time=99000.0))]
    result = match_qar_to_flights([_meta(landing_timestamp=1020.0)], qar)
    assert result.truths == []
    assert result.unmatched_flights == ["F1"]


def _qar_kwargs(**overrides) -> dict:
    kwargs = dict(
        registration="N12345",
        landing_time=1018.0,
        destination_actual_icao="KJFK",
        destination_actual_runway="04L",
        longitude_at_touchdown=THR_LON + 0.004,
        latitude_at_touchdown=THR_LAT + 0.004,
        aircraft_type="B738",
        HEXID="A1B2C3",
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# End-to-end: files -> flights -> matched truth -> pipeline-consumable
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_files_to_matched_flights_end_to_end(tmp_path):
    adsb_path = tmp_path / "adsb.csv"
    qar_path = tmp_path / "qar.csv"
    supp_path = tmp_path / "runways.csv"

    pd.DataFrame(_adsb_rows(n=10)).to_csv(adsb_path, index=False)
    pd.DataFrame([_qar_row(landing_time=1035.0)]).to_csv(qar_path, index=False)
    pd.DataFrame([{
        "airport": "KJFK", "runway": "04L", "heading_deg": HEADING,
        "elevation_m": 3.4, "elevation_datum": "MSL", "width_m": 61.0,
    }]).to_csv(supp_path, index=False)

    supplement = read_runway_supplement(supp_path)
    adsb = read_adsb_flights(adsb_path, supplement=supplement)
    qar = read_qar_file(qar_path)
    match = match_qar_to_flights(adsb.metas, qar)

    assert len(adsb.flights) == 1
    assert len(match.truths) == 1
    flight, truth = adsb.flights[0], match.truths[0]
    assert truth.flight_id == flight.flight_id
    # The record satisfies the pipeline contract: distinct sorted timebases,
    # SI altitudes, knots groundspeed, and a validated runway reference.
    assert np.all(np.diff(flight.position_times) > 0)
    assert np.all(np.diff(flight.velocity_times) > 0)
    assert math.isfinite(flight.runway.length_m)
