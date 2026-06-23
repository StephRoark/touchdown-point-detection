"""Unit tests for the pipeline data models (Task 2).

These tests instantiate every dataclass with representative values and assert
field round-trips, confirm the abstract estimator/fusion interfaces raise
``NotImplementedError``, and confirm the ``FailureReason`` enum members exist.
"""

import dataclasses

import numpy as np
import pytest

from tdz.config.models import LeverArm, SourceCapability
from tdz.models import (
    AireonMessage,
    BaseEstimator,
    FR24Record,
    FailureReason,
    FlightRecord,
    FusedEstimate,
    FusionEnsemble,
    QARTruthRecord,
    RunwayReference,
    TDEstimate,
    TouchdownResult,
    ValidationMetrics,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def make_runway() -> RunwayReference:
    return RunwayReference(
        threshold_lat=40.639801,
        threshold_lon=-73.778900,
        heading_deg=43.0,
        elevation_m=3.0,
        elevation_datum="MSL",
        geoid_undulation_m=-33.0,
        length_m=3460.0,
        width_m=45.0,
        displaced=False,
    )


def make_lever_arm(is_default: bool = False) -> LeverArm:
    return LeverArm(
        icao_type="B738",
        vertical_offset_m=4.2,
        longitudinal_offset_m=12.5,
        nominal_touchdown_pitch_deg=5.5,
        aircraft_class="narrowbody",
        is_class_default=is_default,
    )


# ---------------------------------------------------------------------------
# Estimator contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_td_estimate_round_trip():
    est = TDEstimate(
        t_td=1_700_000_123.5,
        sigma_t=0.8,
        confidence="normal",
        diagnostics={"breakpoint": 1_700_000_123.4},
        method_name="decel_knee",
    )
    assert est.t_td == 1_700_000_123.5
    assert est.sigma_t == 0.8
    assert est.confidence == "normal"
    assert est.diagnostics["breakpoint"] == 1_700_000_123.4
    assert est.method_name == "decel_knee"


@pytest.mark.unit
def test_base_estimator_abstract_methods_raise():
    estimator = BaseEstimator()
    with pytest.raises(NotImplementedError):
        estimator.estimate(flight=None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        estimator.name()


# ---------------------------------------------------------------------------
# Runway / lever arm
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runway_reference_round_trip():
    rwy = make_runway()
    assert rwy.threshold_lat == pytest.approx(40.639801)
    assert rwy.elevation_datum == "MSL"
    assert rwy.geoid_undulation_m == -33.0
    assert rwy.displaced is False


@pytest.mark.unit
def test_lever_arm_round_trip_and_default_flag():
    la = make_lever_arm()
    assert la.icao_type == "B738"
    assert la.vertical_offset_m == 4.2
    assert la.longitudinal_offset_m == 12.5
    assert la.nominal_touchdown_pitch_deg == 5.5
    assert la.aircraft_class == "narrowbody"
    # is_class_default defaults to False
    assert la.is_class_default is False
    assert make_lever_arm(is_default=True).is_class_default is True


@pytest.mark.unit
def test_source_capability_round_trip():
    cap = SourceCapability(
        source="fr24",
        has_geometric_altitude=False,
        samples_are_raw=False,
        async_timestamps=False,
    )
    assert cap.source == "fr24"
    assert cap.has_geometric_altitude is False
    assert cap.samples_are_raw is False
    assert cap.async_timestamps is False


# ---------------------------------------------------------------------------
# Flight record
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flight_record_round_trip_and_optional_defaults():
    pos_t = np.array([0.0, 4.0, 8.0])
    vel_t = np.array([1.0, 5.0, 9.0])
    fr = FlightRecord(
        flight_id="ABC123",
        aircraft_type="B738",
        ads_b_source="aireon",
        position_times=pos_t,
        velocity_times=vel_t,
        latitudes=np.array([40.0, 40.001, 40.002]),
        longitudes=np.array([-73.0, -73.001, -73.002]),
        geometric_altitudes=np.array([100.0, 60.0, 5.0]),
        barometric_altitudes=np.array([110.0, 70.0, 15.0]),
        groundspeeds=np.array([140.0, 130.0, 70.0]),
        tracks=np.array([43.0, 43.0, 43.0]),
        baro_vertical_rates=np.array([-700.0, -300.0, np.nan]),
        on_ground_flags=np.array([False, False, True]),
        on_ground_transition_time=8.0,
        runway=make_runway(),
    )
    assert fr.flight_id == "ABC123"
    assert fr.ads_b_source == "aireon"
    np.testing.assert_array_equal(fr.position_times, pos_t)
    np.testing.assert_array_equal(fr.velocity_times, vel_t)
    assert fr.on_ground_transition_time == 8.0
    assert fr.runway.length_m == 3460.0
    # Module-3-populated derived fields default to None
    assert fr.smoothed_deceleration is None
    assert fr.smoothed_jerk is None
    assert fr.derivative_uncertainties is None
    assert fr.distance_to_threshold is None
    assert fr.time_deltas is None


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fused_estimate_round_trip():
    sub = TDEstimate(1.0, 0.5, "normal", {}, "pelt")
    fused = FusedEstimate(
        t_td=1.0,
        sigma_t=0.4,
        ci_90_lower=0.3,
        ci_90_upper=1.7,
        confidence="normal",
        reason_code=None,
        contributing_estimators=["pelt", "decel_knee"],
        excluded_estimators=["flare_crossing"],
        per_estimator_results={"pelt": sub},
    )
    assert fused.ci_90_lower < fused.t_td < fused.ci_90_upper
    assert fused.reason_code is None
    assert fused.contributing_estimators == ["pelt", "decel_knee"]
    assert fused.per_estimator_results["pelt"] is sub


@pytest.mark.unit
def test_fusion_ensemble_abstract_method_raises():
    ensemble = FusionEnsemble()
    with pytest.raises(NotImplementedError):
        ensemble.fuse(estimates=[], context=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Output record
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_touchdown_result_round_trip():
    la = make_lever_arm()
    result = TouchdownResult(
        flight_id="ABC123",
        aircraft_type="B738",
        ads_b_source="aireon",
        touchdown_time=1_700_000_123.5,
        along_runway_distance_ft=1500.0,
        lateral_offset_ft=12.0,
        groundspeed_at_touchdown_kt=135.0,
        time_ci_90_lower=1_700_000_122.7,
        time_ci_90_upper=1_700_000_124.3,
        distance_ci_90_lower_ft=1350.0,
        distance_ci_90_upper_ft=1650.0,
        speed_ci_90_lower_kt=132.0,
        speed_ci_90_upper_kt=138.0,
        trajectory_type="completed-landing",
        confidence="normal",
        reason_code=None,
        contributing_estimators=["decel_knee", "pelt"],
        excluded_estimators=[],
        physics_anchor_t_td=1_700_000_123.6,
        physics_anchor_diagnostics={"segments": 2},
        lever_arm_used=la,
        lever_arm_missing=False,
        assumed_touchdown_pitch_deg=5.5,
        geometric_altitude_available=True,
        runway_elevation_datum="MSL",
        suspected_wrong_runway=False,
        out_of_bounds=False,
        data_version="2024.1",
        code_commit="abc1234",
        config_hash="deadbeef",
        model_artifact_hash=None,
    )
    assert result.along_runway_distance_ft == 1500.0
    assert result.confidence == "normal"
    assert result.reason_code is None
    assert result.lever_arm_used is la
    assert result.lever_arm_missing is False
    assert result.geometric_altitude_available is True
    assert result.model_artifact_hash is None


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_aireon_message_round_trip_position_and_velocity():
    pos = AireonMessage(
        flight_id="ABC123",
        message_type="position",
        timestamp=100.0,
        latitude=40.0,
        longitude=-73.0,
        geometric_altitude_m=60.0,
        barometric_altitude_m=70.0,
        on_ground=False,
    )
    assert pos.message_type == "position"
    assert pos.latitude == 40.0
    # Velocity fields default to None on a position message
    assert pos.groundspeed_kt is None

    vel = AireonMessage(
        flight_id="ABC123",
        message_type="velocity",
        timestamp=101.0,
        groundspeed_kt=130.0,
        track_deg=43.0,
        baro_vertical_rate_ftmin=-300.0,
    )
    assert vel.message_type == "velocity"
    assert vel.groundspeed_kt == 130.0
    assert vel.latitude is None


@pytest.mark.unit
def test_fr24_record_round_trip():
    rec = FR24Record(
        flight_id="ABC123",
        timestamp=100.0,
        latitude=40.0,
        longitude=-73.0,
        altitude_m=70.0,
        altitude_kind="barometric",
        groundspeed_kt=130.0,
        track_deg=43.0,
        on_ground=False,
    )
    assert rec.altitude_kind == "barometric"
    assert rec.on_ground is False
    # Optional vertical rate defaults to None
    assert rec.vertical_rate_ftmin is None


# ---------------------------------------------------------------------------
# QAR truth & validation metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_qar_truth_record_round_trip():
    rec = QARTruthRecord(
        flight_id="ABC123",
        touchdown_time_qar=1_700_000_124.0,
        touchdown_lat=40.6402,
        touchdown_lon=-73.7785,
        clock_offset_estimate=0.5,
        clock_offset_quality="good",
        aircraft_type="B738",
        runway_id="04L",
        airport_id="KJFK",
        tail_number="N12345",
    )
    assert rec.clock_offset_quality == "good"
    assert rec.tail_number == "N12345"


@pytest.mark.unit
def test_validation_metrics_round_trip_and_default_stratum():
    m = ValidationMetrics(
        n_flights=500,
        distance_rmse_ft=210.0,
        distance_median_abs_error_ft=120.0,
        distance_iqr_ft=(80.0, 180.0),
        distance_p95_abs_error_ft=420.0,
        distance_p99_abs_error_ft=610.0,
        distance_p95_long_side_ft=380.0,
        distance_median_signed_error_ft=10.0,
        time_rmse_s=0.9,
        time_median_abs_error_s=0.6,
        baseline_rmse_ft=350.0,
        improvement_pct=40.0,
        ci_90_coverage=0.91,
    )
    assert m.distance_iqr_ft == (80.0, 180.0)
    assert m.ci_90_coverage == pytest.approx(0.91)
    # stratum_key defaults to None
    assert m.stratum_key is None


# ---------------------------------------------------------------------------
# Failure reason enum
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_failure_reason_members_exist():
    expected = {
        # No-estimate reasons
        "INSUFFICIENT_SAMPLES": "insufficient_samples",
        "NO_GROUNDSPEED": "no_groundspeed_data",
        "GAP_SPANS_TOUCHDOWN": "gap_spans_touchdown",
        "EXCESSIVE_EXCLUSIONS": "excessive_exclusions",
        "ALL_ESTIMATORS_FAILED": "all_estimators_failed",
        "INVALID_RUNWAY_REF": "invalid_runway_reference",
        "GO_AROUND": "go_around",
        "TOUCH_AND_GO": "touch_and_go",
        # Low-confidence reasons
        "SPARSE_NEAR_TD": "sparse_near_touchdown",
        "WIDE_CONFIDENCE_INTERVAL": "wide_ci",
        "MISSING_VERTICAL_RATE": "missing_vertical_rate",
        "MISSING_LEVER_ARM": "missing_lever_arm",
        "ESTIMATOR_DISAGREEMENT": "estimator_disagreement",
        "OUT_OF_BOUNDS_POSITION": "out_of_bounds_position",
        "DEGRADED_INTERPOLATION": "degraded_interpolation",
        "INSUFFICIENT_FLARE_SAMPLES": "insufficient_flare",
        "NO_GROUND_ROLL_CONFIRMATION": "no_ground_roll",
        "GEOMETRIC_ALT_UNAVAILABLE": "geometric_alt_unavailable",
        "SUSPECTED_WRONG_RUNWAY": "suspected_wrong_runway",
        "CLOCK_OFFSET_EXCEEDED": "clock_offset_exceeded",
        "DATUM_UNRESOLVED": "datum_unresolved",
    }
    for name, value in expected.items():
        assert hasattr(FailureReason, name), f"missing FailureReason.{name}"
        assert getattr(FailureReason, name).value == value


@pytest.mark.unit
def test_failure_reason_missing_lever_arm_defined_once():
    # The design listed MISSING_LEVER_ARM twice; ensure only one member maps to
    # the value (no alias duplication producing surprises).
    members = [r for r in FailureReason if r.value == "missing_lever_arm"]
    assert len(members) == 1


@pytest.mark.unit
def test_models_are_dataclasses():
    for cls in (
        TDEstimate,
        RunwayReference,
        FlightRecord,
        FusedEstimate,
        TouchdownResult,
        AireonMessage,
        FR24Record,
        LeverArm,
        SourceCapability,
        QARTruthRecord,
        ValidationMetrics,
    ):
        assert dataclasses.is_dataclass(cls)
