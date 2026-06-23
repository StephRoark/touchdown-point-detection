"""Tests for the raw ADS-B / QAR input schemas (tdz.io.raw_schema).

These confirm the schema metadata is internally consistent and that the typed
row dataclasses cover exactly the delivered columns, so ingest (Task 9) can rely
on the schema as the single source of truth for raw layout.
"""

from __future__ import annotations

import dataclasses

import pytest

from tdz.io import (
    ADSB_FIELDS,
    ADSB_TO_FLIGHTRECORD,
    JOIN_KEYS,
    QAR_FIELDS,
    QAR_TO_TRUTHRECORD,
    ADSBRecord,
    QARRecord,
    adsb_columns,
    adsb_pandas_dtypes,
    qar_columns,
    qar_pandas_dtypes,
)

# The authoritative column lists provided by the data owner.
EXPECTED_ADSB_COLUMNS = [
    "flight_id", "timestamp", "hexid", "time_position", "time_velocity",
    "latitude", "longitude", "ground_speed", "airport", "runway", "on_ground",
    "landing_timestamp", "landing_latitude", "landing_longitude",
    "aircraft_registration", "runway_length", "model_icao",
    "aircraft_model_number_and_subtype", "threshold_latitude",
    "threshold_longitude", "threshold_diplacement_length",
    "opposite_displacement_length", "lda_day", "ground_speed_airborne",
    "ground_speed_surface", "barometric_altitude", "barometric_vertical_rate",
    "geometric_height", "selected_altitude", "track",
]

EXPECTED_QAR_COLUMNS = [
    "registration", "landing_time", "destination_actual_icao",
    "destination_actual_runway", "destination_runway_length",
    "longitude_at_touchdown", "latitude_at_touchdown",
    "destination_runway_threshold_latitude",
    "destination_runway_threshold_longitude", "deceleration_at_60kt",
    "cas_touchdown", "gs_touchdown", "aircraft_type", "aircraft_subtype",
    "runway_length", "HEXID",
]

_VALID_DTYPES = {"float64", "string", "boolean", "Int64"}
_VALID_CONSTANCY = {"per_sample", "per_flight"}


@pytest.mark.unit
def test_adsb_columns_match_delivered_layout():
    assert adsb_columns() == EXPECTED_ADSB_COLUMNS


@pytest.mark.unit
def test_qar_columns_match_delivered_layout():
    assert qar_columns() == EXPECTED_QAR_COLUMNS


@pytest.mark.unit
def test_no_duplicate_columns():
    assert len(adsb_columns()) == len(set(adsb_columns()))
    assert len(qar_columns()) == len(set(qar_columns()))


@pytest.mark.unit
@pytest.mark.parametrize("fields", [ADSB_FIELDS, QAR_FIELDS])
def test_field_specs_are_well_formed(fields):
    for spec in fields:
        assert spec.dtype in _VALID_DTYPES, f"{spec.name}: bad dtype {spec.dtype}"
        assert spec.constancy in _VALID_CONSTANCY, f"{spec.name}: bad constancy"
        assert spec.description, f"{spec.name}: missing description"
        assert spec.unit, f"{spec.name}: missing unit"


@pytest.mark.unit
def test_pandas_dtype_maps_cover_all_columns():
    assert set(adsb_pandas_dtypes()) == set(adsb_columns())
    assert set(qar_pandas_dtypes()) == set(qar_columns())


@pytest.mark.unit
def test_record_dataclasses_field_names_match_schema():
    adsb_record_fields = {f.name for f in dataclasses.fields(ADSBRecord)}
    qar_record_fields = {f.name for f in dataclasses.fields(QARRecord)}
    assert adsb_record_fields == set(adsb_columns())
    assert qar_record_fields == set(qar_columns())


@pytest.mark.unit
def test_displacement_typo_preserved():
    # The source column is misspelled 'diplacement'; the schema must match it.
    assert "threshold_diplacement_length" in adsb_columns()
    assert hasattr(ADSBRecord("f", "h", "ap", "rw", "reg", "B738", 0.0),
                   "threshold_diplacement_length")


@pytest.mark.unit
def test_mappings_reference_real_columns_and_fields():
    # Mapping keys must be real raw columns.
    assert set(ADSB_TO_FLIGHTRECORD).issubset(set(adsb_columns()))
    assert set(QAR_TO_TRUTHRECORD).issubset(set(qar_columns()))
    # Join keys reference one column on each side.
    for adsb_col, qar_col in JOIN_KEYS.values():
        assert adsb_col in adsb_columns()
        assert qar_col in qar_columns()


@pytest.mark.unit
def test_records_instantiate_with_minimal_required_fields():
    adsb = ADSBRecord(
        flight_id="ABC", hexid="A1B2C3", airport="KJFK", runway="04L",
        aircraft_registration="N123", model_icao="B738", timestamp=100.0,
    )
    assert adsb.latitude is None and adsb.on_ground is None

    qar = QARRecord(
        registration="N123", landing_time=100.0,
        destination_actual_icao="KJFK", destination_actual_runway="04L",
        longitude_at_touchdown=-73.78, latitude_at_touchdown=40.64,
        aircraft_type="B737",
    )
    assert qar.HEXID is None and qar.gs_touchdown is None
