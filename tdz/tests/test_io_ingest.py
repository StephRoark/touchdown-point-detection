"""Tests for the ingest + source-capability gating modules (Task 9.1, 9.2).

Covers Property 20 (Source-Capability Estimator Gating) plus known-answer unit
tests: Aireon async parse preserves the separate position/velocity timebases;
FR24 parse sets co-timed arrays and records the source; the FR24 (no geometric
altitude) source excludes ``flare_crossing`` (with GEOMETRIC_ALT_UNAVAILABLE)
while a speed/position estimate path remains; barometric altitude is never
substituted into the geometric field; and the optional QAR-by-HEXID join.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.config.models import SourceCapability
from tdz.io import (
    GEOMETRIC_ALTITUDE_ESTIMATORS,
    find_qar_truth,
    gate_estimators,
    parse_aireon_messages,
    parse_fr24_records,
)
from tdz.io.raw_schema import QARRecord
from tdz.models import AireonMessage, FailureReason, FR24Record, RunwayReference

_ENABLED = (
    "decel_knee",
    "flare_crossing",
    "imm_rts",
    "jerk_onset",
    "pelt",
    "lightgbm",
    "sequence_model",
)


def _runway() -> RunwayReference:
    return RunwayReference(
        threshold_lat=33.94,
        threshold_lon=-118.40,
        heading_deg=250.0,
        elevation_m=38.0,
        elevation_datum="MSL",
        geoid_undulation_m=-35.0,
        length_m=3000.0,
        width_m=45.0,
        displaced=False,
    )


# ---------------------------------------------------------------------------
# Property 20: Source-Capability Estimator Gating
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    has_geo=st.booleans(),
    samples_raw=st.booleans(),
    enabled=st.lists(
        st.sampled_from(sorted(set(_ENABLED) | GEOMETRIC_ALTITUDE_ESTIMATORS)),
        min_size=1,
        max_size=10,
        unique=True,
    ),
)
def test_source_capability_gating(has_geo, samples_raw, enabled):
    """Feature: touchdown-point-detection, Property 20: Source-Capability Estimator Gating

    For a source whose capability reports no geometric altitude, every enabled
    geometric-altitude estimator is excluded with GEOMETRIC_ALT_UNAVAILABLE and
    none survive into the eligible set; non-geometric estimators always remain
    eligible. The samples-independence flag mirrors ``samples_are_raw``
    (Req 8.5-8.8).
    """
    cap = SourceCapability(
        source="fr24" if not has_geo else "aireon",
        has_geometric_altitude=has_geo,
        samples_are_raw=samples_raw,
        async_timestamps=True,  # value irrelevant to gating
    )
    gating = gate_estimators(cap, enabled)

    geom_in_enabled = [e for e in enabled if e in GEOMETRIC_ALTITUDE_ESTIMATORS]
    nongeom_in_enabled = [e for e in enabled if e not in GEOMETRIC_ALTITUDE_ESTIMATORS]

    if not has_geo:
        # All geometric estimators excluded with the right reason code.
        for e in geom_in_enabled:
            assert e in gating.excluded_estimators
            assert e not in gating.eligible_estimators
            assert gating.reason_for(e) is FailureReason.GEOMETRIC_ALT_UNAVAILABLE
    else:
        # With geometric altitude present nothing is source-excluded here.
        assert gating.excluded_estimators == ()
        for e in geom_in_enabled:
            assert e in gating.eligible_estimators

    # Non-geometric estimators are always eligible.
    for e in nongeom_in_enabled:
        assert e in gating.eligible_estimators

    # Eligible + excluded partition the enabled set with no overlap.
    assert set(gating.eligible_estimators) | set(gating.excluded_estimators) == set(
        enabled
    )
    assert set(gating.eligible_estimators) & set(gating.excluded_estimators) == set()

    # Provider-interpolated samples are flagged non-independent.
    assert gating.samples_independent is bool(samples_raw)


# ---------------------------------------------------------------------------
# Aireon async parse: separate timebases preserved
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aireon_parse_preserves_separate_timebases():
    """Aireon parse keeps distinct position/velocity timebases (Req 8.3)."""
    msgs = [
        AireonMessage("F1", "position", 0.0, latitude=10.0, longitude=20.0,
                      geometric_altitude_m=300.0, barometric_altitude_m=290.0,
                      on_ground=False),
        AireonMessage("F1", "position", 4.0, latitude=10.01, longitude=20.0,
                      geometric_altitude_m=200.0, barometric_altitude_m=190.0,
                      on_ground=False),
        AireonMessage("F1", "position", 8.0, latitude=10.02, longitude=20.0,
                      geometric_altitude_m=100.0, barometric_altitude_m=95.0,
                      on_ground=True),
        AireonMessage("F1", "velocity", 1.5, groundspeed_kt=140.0, track_deg=10.0,
                      baro_vertical_rate_ftmin=-600.0),
        AireonMessage("F1", "velocity", 5.5, groundspeed_kt=120.0, track_deg=10.0,
                      baro_vertical_rate_ftmin=-500.0),
    ]
    rec = parse_aireon_messages(msgs, runway=_runway(), aircraft_type="B738")

    assert rec.ads_b_source == "aireon"
    assert rec.flight_id == "F1"
    assert rec.aircraft_type == "B738"
    # Distinct timebases, not merged.
    assert rec.position_times.tolist() == [0.0, 4.0, 8.0]
    assert rec.velocity_times.tolist() == [1.5, 5.5]
    assert rec.position_times.size != rec.velocity_times.size
    # Position channels track the position timebase.
    assert rec.geometric_altitudes.tolist() == [300.0, 200.0, 100.0]
    assert rec.groundspeeds.tolist() == [140.0, 120.0]
    # On-ground transition detected at t=8.
    assert rec.on_ground_transition_time == 8.0


@pytest.mark.unit
def test_aireon_parse_sorts_and_fills_missing_with_nan():
    """Out-of-order messages are time-sorted; missing fields become NaN."""
    msgs = [
        AireonMessage("F2", "velocity", 6.0, groundspeed_kt=100.0, track_deg=5.0),
        AireonMessage("F2", "velocity", 2.0, groundspeed_kt=None, track_deg=None),
        AireonMessage("F2", "position", 4.0, latitude=1.0, longitude=2.0),
    ]
    rec = parse_aireon_messages(msgs, runway=_runway(), aircraft_type="B77W")
    assert rec.velocity_times.tolist() == [2.0, 6.0]
    assert np.isnan(rec.groundspeeds[0])
    assert rec.groundspeeds[1] == 100.0
    # Position message had no geometric altitude -> NaN.
    assert np.isnan(rec.geometric_altitudes[0])


@pytest.mark.unit
def test_aireon_parse_rejects_unknown_message_type():
    with pytest.raises(ValueError, match="unknown Aireon message_type"):
        parse_aireon_messages(
            [AireonMessage("F", "weather", 0.0)],
            runway=_runway(),
            aircraft_type="B738",
        )


@pytest.mark.unit
def test_aireon_parse_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        parse_aireon_messages([], runway=_runway(), aircraft_type="B738")


# ---------------------------------------------------------------------------
# FR24 co-timed parse: co-timed arrays + recorded source + altitude routing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fr24_parse_sets_cotimed_arrays_and_records_source():
    """FR24 parse sets position_times == velocity_times and records the source."""
    rows = [
        FR24Record("G1", 0.0, 10.0, 20.0, 300.0, "barometric", 150.0, 90.0, False),
        FR24Record("G1", 4.0, 10.0, 20.01, 200.0, "barometric", 130.0, 90.0, False),
        FR24Record("G1", 8.0, 10.0, 20.02, 100.0, "barometric", 110.0, 90.0, True),
    ]
    rec = parse_fr24_records(rows, runway=_runway(), aircraft_type="A320")

    assert rec.ads_b_source == "flightradar24"
    # Co-timed: the two timebases hold identical values.
    assert np.array_equal(rec.position_times, rec.velocity_times)
    assert rec.position_times.tolist() == [0.0, 4.0, 8.0]
    # ...but are separate array objects (no aliasing).
    assert rec.position_times is not rec.velocity_times
    # Barometric altitude routed to the baro field; geometric stays all-NaN.
    assert rec.barometric_altitudes.tolist() == [300.0, 200.0, 100.0]
    assert np.all(np.isnan(rec.geometric_altitudes))


@pytest.mark.unit
def test_fr24_geometric_kind_routes_to_geometric_field():
    """A geometric-tagged FR24 row populates the geometric field, not baro."""
    rows = [
        FR24Record("G2", 0.0, 1.0, 2.0, 305.0, "geometric", 150.0, 90.0, False),
    ]
    rec = parse_fr24_records(rows, runway=_runway(), aircraft_type="A320")
    assert rec.geometric_altitudes.tolist() == [305.0]
    assert np.all(np.isnan(rec.barometric_altitudes))


@pytest.mark.unit
def test_fr24_no_geometric_excludes_flare_but_keeps_speed_path():
    """FR24 (no geometric altitude) -> flare_crossing excluded, speed path remains.

    Mirrors the design edge case: a FlightRadar24-style source disables the
    vertical estimators yet still has a speed/position estimate path (decel_knee,
    jerk_onset, pelt, learned). Barometric altitude is never placed into the
    geometric field.
    """
    rows = [
        FR24Record("G3", float(t), 1.0, 2.0 + 0.001 * t, 300.0 - 20.0 * t,
                   "barometric", 150.0 - 4.0 * t, 90.0, t >= 8)
        for t in range(0, 12, 4)
    ]
    rec = parse_fr24_records(rows, runway=_runway(), aircraft_type="A320")
    assert np.all(np.isnan(rec.geometric_altitudes))  # never substituted (Req 8.8)

    cap = SourceCapability("fr24", has_geometric_altitude=False,
                           samples_are_raw=False, async_timestamps=False)
    gating = gate_estimators(cap, _ENABLED)

    assert "flare_crossing" in gating.excluded_estimators
    assert gating.reason_for("flare_crossing") is FailureReason.GEOMETRIC_ALT_UNAVAILABLE
    # A speed/position estimate path remains eligible.
    for survivor in ("decel_knee", "jerk_onset", "pelt", "lightgbm"):
        assert survivor in gating.eligible_estimators
    # Provider-interpolated -> samples must not be treated as independent.
    assert gating.samples_independent is False


# ---------------------------------------------------------------------------
# Optional QAR-by-HEXID/registration join helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_qar_truth_by_hexid_then_registration():
    qar = [
        QARRecord("N111AA", 100.0, "KLAX", "25L", -118.4, 33.9, "B738", HEXID="ABC123"),
        QARRecord("N222BB", 200.0, "KSFO", "28R", -122.4, 37.6, "B739", HEXID="DEF456"),
    ]
    # HEXID match takes precedence.
    hit = find_qar_truth(qar, hexid="DEF456")
    assert hit is not None and hit.registration == "N222BB"
    # Registration fallback.
    hit2 = find_qar_truth(qar, registration="N111AA")
    assert hit2 is not None and hit2.HEXID == "ABC123"
    # No match.
    assert find_qar_truth(qar, hexid="ZZZ999") is None
    # Must supply at least one key.
    with pytest.raises(ValueError):
        find_qar_truth(qar)
