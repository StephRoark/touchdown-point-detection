"""Tests for runway-reference validation (Task 4.3 / 4.4).

Covers Property 18 (Runway Reference Validation) plus edge cases at the bounds.
"""

import dataclasses
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.geo import InvalidRunwayReferenceError, validate_runway_reference
from tdz.models import FailureReason, RunwayReference


def _valid_runway() -> RunwayReference:
    return RunwayReference(
        threshold_lat=40.639801,
        threshold_lon=-73.778900,
        heading_deg=43.0,
        elevation_m=3.0,
        elevation_datum="HAE",
        geoid_undulation_m=-30.0,
        length_m=3000.0,
        width_m=45.0,
        displaced=False,
    )


# Validated fields and a set of out-of-bounds / invalid replacement values.
_OUT_OF_BOUNDS = {
    "threshold_lat": [90.001, -90.001, 1000.0, None, math.nan],
    "threshold_lon": [180.001, -180.001, 360.0, None, math.nan],
    "heading_deg": [-0.001, 360.001, 720.0, None, math.nan],
    "elevation_m": [-500.001, 10000.001, None, math.nan],
    "length_m": [0.0, -1.0, 6000.001, None, math.nan],
    "width_m": [0.0, -5.0, 100.001, None, math.nan],
}

_FIELDS = sorted(_OUT_OF_BOUNDS.keys())


# ---------------------------------------------------------------------------
# Property 18: Runway Reference Validation
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    field=st.sampled_from(_FIELDS),
    bad_index=st.integers(min_value=0, max_value=10),
)
def test_out_of_bounds_field_rejected(field, bad_index):
    """Feature: touchdown-point-detection, Property 18: Runway Reference Validation

    A runway reference with any single required field forced out of bounds
    (or None/NaN) is rejected with an error naming that field and carrying the
    INVALID_RUNWAY_REF reason code.
    """
    bad_values = _OUT_OF_BOUNDS[field]
    bad_value = bad_values[bad_index % len(bad_values)]
    runway = dataclasses.replace(_valid_runway(), **{field: bad_value})

    with pytest.raises(InvalidRunwayReferenceError) as excinfo:
        validate_runway_reference(runway)

    assert excinfo.value.field == field
    assert field in str(excinfo.value)
    assert excinfo.value.reason_code is FailureReason.INVALID_RUNWAY_REF


@pytest.mark.property
@given(
    lat=st.floats(min_value=-90.0, max_value=90.0),
    lon=st.floats(min_value=-180.0, max_value=180.0),
    heading=st.floats(min_value=0.0, max_value=360.0),
    elevation=st.floats(min_value=-500.0, max_value=10000.0),
    length=st.floats(min_value=1e-6, max_value=6000.0),
    width=st.floats(min_value=1e-6, max_value=100.0),
)
def test_in_bounds_reference_passes(lat, lon, heading, elevation, length, width):
    """Feature: touchdown-point-detection, Property 18: Runway Reference Validation

    A fully in-bounds runway reference validates without raising.
    """
    runway = dataclasses.replace(
        _valid_runway(),
        threshold_lat=lat,
        threshold_lon=lon,
        heading_deg=heading,
        elevation_m=elevation,
        length_m=length,
        width_m=width,
    )
    # Should not raise.
    assert validate_runway_reference(runway) is None


# ---------------------------------------------------------------------------
# Edge tests at the bounds (Task 4.4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_valid_reference_passes():
    assert validate_runway_reference(_valid_runway()) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold_lat", 90.0),
        ("threshold_lat", -90.0),
        ("threshold_lon", 180.0),
        ("threshold_lon", -180.0),
        ("heading_deg", 0.0),
        ("heading_deg", 360.0),
        ("elevation_m", -500.0),
        ("elevation_m", 10000.0),
        ("length_m", 6000.0),
        ("width_m", 100.0),
    ],
)
def test_inclusive_bounds_accepted(field, value):
    """Inclusive bound values are accepted."""
    runway = dataclasses.replace(_valid_runway(), **{field: value})
    assert validate_runway_reference(runway) is None


@pytest.mark.unit
@pytest.mark.parametrize("field", _FIELDS)
def test_none_field_rejected_naming_field(field):
    """A missing (None) required field is rejected, naming the field."""
    runway = dataclasses.replace(_valid_runway(), **{field: None})
    with pytest.raises(InvalidRunwayReferenceError) as excinfo:
        validate_runway_reference(runway)
    assert excinfo.value.field == field
    assert excinfo.value.reason_code is FailureReason.INVALID_RUNWAY_REF


@pytest.mark.unit
@pytest.mark.parametrize("field", _FIELDS)
def test_nan_field_rejected_naming_field(field):
    """A NaN required field is rejected, naming the field."""
    runway = dataclasses.replace(_valid_runway(), **{field: math.nan})
    with pytest.raises(InvalidRunwayReferenceError) as excinfo:
        validate_runway_reference(runway)
    assert excinfo.value.field == field


@pytest.mark.unit
def test_zero_length_rejected():
    """Length must be strictly positive (Req 11.5)."""
    runway = dataclasses.replace(_valid_runway(), length_m=0.0)
    with pytest.raises(InvalidRunwayReferenceError) as excinfo:
        validate_runway_reference(runway)
    assert excinfo.value.field == "length_m"


@pytest.mark.unit
def test_zero_width_rejected():
    """Width must be strictly positive (Req 11.5)."""
    runway = dataclasses.replace(_valid_runway(), width_m=0.0)
    with pytest.raises(InvalidRunwayReferenceError) as excinfo:
        validate_runway_reference(runway)
    assert excinfo.value.field == "width_m"


@pytest.mark.unit
def test_infinite_elevation_rejected():
    runway = dataclasses.replace(_valid_runway(), elevation_m=math.inf)
    with pytest.raises(InvalidRunwayReferenceError) as excinfo:
        validate_runway_reference(runway)
    assert excinfo.value.field == "elevation_m"
