"""Tests for vertical datum unification (Task 5).

Covers Property 19 (Vertical Datum Consistency / Geoid) plus known-answer and
error edge cases. The core guarantee: an MSL elevation H with undulation N
resolves to H + N, and an equivalent HAE-tagged elevation (H + N) resolves to
the same value -- i.e. MSL and HAE agree *after* the geoid correction, and an
MSL elevation is never compared as HAE without it (Requirement 11.2, 17.2).
"""

import dataclasses
import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.geo import (
    DatumResolver,
    DatumUnresolvedError,
    lookup_geoid_undulation,
    resolve_threshold_elevation_hae,
)
from tdz.models import FailureReason, RunwayReference


def _runway(
    *, elevation_m: float, elevation_datum: str, geoid_undulation_m: float
) -> RunwayReference:
    """Build a runway reference with the given vertical-datum fields."""
    return RunwayReference(
        threshold_lat=40.639801,
        threshold_lon=-73.778900,
        heading_deg=43.0,
        elevation_m=elevation_m,
        elevation_datum=elevation_datum,
        geoid_undulation_m=geoid_undulation_m,
        length_m=3000.0,
        width_m=45.0,
        displaced=False,
    )


# Tight tolerance: the conversion is exact arithmetic (no geodesy involved).
_TOL_M = 1e-6


# ---------------------------------------------------------------------------
# Property 19: Vertical Datum Consistency (Geoid)
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    true_msl_elevation_m=st.floats(min_value=-400.0, max_value=4000.0),
    undulation_m=st.floats(min_value=-100.0, max_value=100.0),
)
def test_msl_and_hae_agree_after_correction(true_msl_elevation_m, undulation_m):
    """Feature: touchdown-point-detection, Property 19: Vertical Datum Consistency (Geoid)

    For a true MSL elevation H and geoid undulation N, a runway tagged MSL with
    geoid_undulation_m=N resolves to H + N (the HAE datum), and an equivalent
    runway tagged HAE with elevation_m = H + N resolves to the SAME HAE value.
    An MSL elevation is never returned as HAE without the geoid correction.
    """
    expected_hae = true_msl_elevation_m + undulation_m

    msl_runway = _runway(
        elevation_m=true_msl_elevation_m,
        elevation_datum="MSL",
        geoid_undulation_m=undulation_m,
    )
    hae_runway = _runway(
        elevation_m=expected_hae,
        elevation_datum="HAE",
        geoid_undulation_m=undulation_m,  # present but must be ignored for HAE
    )

    resolved_from_msl = resolve_threshold_elevation_hae(msl_runway)
    resolved_from_hae = resolve_threshold_elevation_hae(hae_runway)

    assert resolved_from_msl == pytest.approx(expected_hae, abs=_TOL_M)
    assert resolved_from_hae == pytest.approx(expected_hae, abs=_TOL_M)
    assert resolved_from_msl == pytest.approx(resolved_from_hae, abs=_TOL_M)

    # The MSL elevation must NOT be returned uncorrected (unless N happens to be
    # ~0). This is the "do not compare MSL as HAE" guarantee (Req 11.2).
    if abs(undulation_m) > _TOL_M:
        assert resolved_from_msl != pytest.approx(true_msl_elevation_m, abs=_TOL_M)


# ---------------------------------------------------------------------------
# Known-answer tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_msl_known_answer_continental_us():
    """MSL 100.0 m with N = -30.0 m -> HAE 70.0 m exactly."""
    runway = _runway(
        elevation_m=100.0, elevation_datum="MSL", geoid_undulation_m=-30.0
    )
    assert resolve_threshold_elevation_hae(runway) == pytest.approx(70.0, abs=_TOL_M)


@pytest.mark.unit
def test_hae_returned_unchanged_undulation_not_applied():
    """An HAE-tagged elevation is returned unchanged; undulation is NOT applied."""
    runway = _runway(
        elevation_m=70.0, elevation_datum="HAE", geoid_undulation_m=-30.0
    )
    # If the undulation were (wrongly) applied, we'd get 40.0; it must stay 70.0.
    assert resolve_threshold_elevation_hae(runway) == pytest.approx(70.0, abs=_TOL_M)


@pytest.mark.unit
def test_positive_undulation_added():
    """A positive undulation is added (h = H + N)."""
    runway = _runway(
        elevation_m=50.0, elevation_datum="MSL", geoid_undulation_m=12.0
    )
    assert resolve_threshold_elevation_hae(runway) == pytest.approx(62.0, abs=_TOL_M)


@pytest.mark.unit
def test_datum_tag_is_case_insensitive():
    """Lowercase datum tags are normalized (defensive against input casing)."""
    runway = _runway(
        elevation_m=100.0, elevation_datum="msl", geoid_undulation_m=-30.0
    )
    assert resolve_threshold_elevation_hae(runway) == pytest.approx(70.0, abs=_TOL_M)


@pytest.mark.unit
def test_resolver_class_matches_function():
    """DatumResolver.resolve matches the module-level convenience function."""
    runway = _runway(
        elevation_m=250.0, elevation_datum="MSL", geoid_undulation_m=-22.5
    )
    resolver = DatumResolver()
    assert resolver.resolve(runway) == pytest.approx(
        resolve_threshold_elevation_hae(runway), abs=_TOL_M
    )


# ---------------------------------------------------------------------------
# Error tests: DATUM_UNRESOLVED
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("bad_undulation", [None, math.nan, math.inf, -math.inf])
def test_msl_missing_or_nonfinite_undulation_raises(bad_undulation):
    """MSL datum with missing/None/NaN/inf undulation and no geoid model raises."""
    runway = _runway(
        elevation_m=100.0,
        elevation_datum="MSL",
        geoid_undulation_m=bad_undulation,
    )
    # Disable the optional pyproj lookup so the supplied-undulation path is the
    # only source; in a grid-less environment this is also the effective state.
    resolver = DatumResolver(allow_geoid_lookup=False)
    with pytest.raises(DatumUnresolvedError) as excinfo:
        resolver.resolve(runway)
    assert excinfo.value.reason_code is FailureReason.DATUM_UNRESOLVED
    assert "undulation" in str(excinfo.value)


@pytest.mark.unit
@pytest.mark.parametrize("bad_datum", ["WGS84", "egm", "", "  ", None, 5])
def test_unrecognized_datum_raises(bad_datum):
    """An unrecognized/missing datum tag with no configured default raises."""
    runway = _runway(
        elevation_m=100.0, elevation_datum="MSL", geoid_undulation_m=-30.0
    )
    runway = dataclasses.replace(runway, elevation_datum=bad_datum)
    with pytest.raises(DatumUnresolvedError) as excinfo:
        resolve_threshold_elevation_hae(runway)
    assert excinfo.value.reason_code is FailureReason.DATUM_UNRESOLVED


@pytest.mark.unit
@pytest.mark.parametrize("bad_elevation", [None, math.nan, math.inf])
def test_nonfinite_elevation_raises(bad_elevation):
    """A missing/non-finite elevation cannot be resolved to a finite HAE value."""
    runway = _runway(
        elevation_m=100.0, elevation_datum="HAE", geoid_undulation_m=-30.0
    )
    runway = dataclasses.replace(runway, elevation_m=bad_elevation)
    with pytest.raises(DatumUnresolvedError) as excinfo:
        resolve_threshold_elevation_hae(runway)
    assert excinfo.value.reason_code is FailureReason.DATUM_UNRESOLVED


# ---------------------------------------------------------------------------
# Configuration-driven datum policy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_untagged_datum_uses_configured_assumption():
    """An untagged elevation is resolved using assume_runway_elevation_datum."""
    runway = _runway(
        elevation_m=100.0, elevation_datum="MSL", geoid_undulation_m=-30.0
    )
    runway = dataclasses.replace(runway, elevation_datum="")

    class _Geodesy:
        geoid_model = "EGM2008"
        assume_runway_elevation_datum = "MSL"

    # With the MSL assumption, the undulation is applied: 100 + (-30) = 70.
    assert resolve_threshold_elevation_hae(runway, _Geodesy()) == pytest.approx(
        70.0, abs=_TOL_M
    )


@pytest.mark.unit
def test_from_config_none_uses_defaults():
    """A None geodesy config resolves a normally-tagged runway via defaults."""
    runway = _runway(
        elevation_m=100.0, elevation_datum="MSL", geoid_undulation_m=-30.0
    )
    assert resolve_threshold_elevation_hae(runway, None) == pytest.approx(
        70.0, abs=_TOL_M
    )


# ---------------------------------------------------------------------------
# Optional pyproj geoid-grid lookup (skipped when grids are unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_geoid_lookup_degrades_without_grids():
    """lookup_geoid_undulation never raises; it returns a float or None."""
    # Regardless of grid availability this must not raise (graceful degradation).
    result = lookup_geoid_undulation(40.639801, -73.778900, geoid_model="EGM2008")
    assert result is None or isinstance(result, float)


@pytest.mark.unit
def test_geoid_grid_lookup_known_region():
    """If EGM2008 grids are present, the CONUS undulation is negative (~ -30 m).

    SKIPPED when the geoid grid is not installed/downloadable, so the suite
    passes in a grid-less sandbox.
    """
    # Near KJFK the EGM2008 undulation is roughly -32 m.
    undulation = lookup_geoid_undulation(40.639801, -73.778900, geoid_model="EGM2008")
    if undulation is None:
        pytest.skip("EGM2008 geoid grids not available in this environment")
    assert -45.0 < undulation < -15.0
