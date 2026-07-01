"""Tests for CI coverage, the cadence-limited floor, and below-target flags (Task 22.4).

Covers empirical 90%-CI coverage assessment on the calibration split for both
the time and distance intervals with the 85-95% acceptance band (Req 4.3, 4.4),
the interpolation-limited error floor imposed by the ADS-B update cadence (Req
13.0, 13.1), and advisory below-target flagging against the provisional accuracy
targets with the >=200-flight gate and no hard-fail behavior (Req 13.2-13.5).
"""

from __future__ import annotations

import math

import pytest
from pyproj import Geod

from tdz.config.schema import (
    ProvisionalAccuracyTargets,
    ValidationConfig,
)
from tdz.models import (
    LeverArm,
    QARTruthRecord,
    RunwayReference,
    TouchdownResult,
    ValidationMetrics,
)
from tdz.uncertainty import M_TO_FT
from tdz.validation import (
    COVERAGE_IN_BAND,
    COVERAGE_OVER,
    COVERAGE_UNDEFINED,
    COVERAGE_UNDER,
    KNOTS_TO_FT_PER_S,
    BelowTargetFlag,
    FlightEvaluation,
    StratifiedMetricsReport,
    StratumResult,
    along_runway_truth_distance_ft,
    assess_coverage,
    cadence_limited_floor_ft,
    characterize_error_floor,
    classify_coverage,
    flag_below_target,
)

_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Builders (mirroring test_validation_metrics conventions)
# ---------------------------------------------------------------------------


def _runway(heading_deg: float = 90.0) -> RunwayReference:
    return RunwayReference(
        threshold_lat=40.0,
        threshold_lon=-75.0,
        heading_deg=heading_deg,
        elevation_m=30.0,
        elevation_datum="HAE",
        geoid_undulation_m=0.0,
        length_m=3500.0,
        width_m=45.0,
        displaced=False,
    )


def _point_along(runway: RunwayReference, along_m: float) -> tuple[float, float]:
    lon2, lat2, _ = _GEOD.fwd(
        runway.threshold_lon, runway.threshold_lat, runway.heading_deg, along_m
    )
    return lat2, lon2


def _lever_arm() -> LeverArm:
    return LeverArm(
        icao_type="B738",
        vertical_offset_m=4.2,
        longitudinal_offset_m=12.5,
        nominal_touchdown_pitch_deg=5.5,
        aircraft_class="narrowbody",
        is_class_default=False,
    )


def _result(
    *,
    flight_id: str = "F1",
    aircraft_type: str = "B738",
    ads_b_source: str = "aireon",
    touchdown_time: float = 1000.0,
    along_runway_distance_ft: float = 1500.0,
    groundspeed_at_touchdown_kt: float = 135.0,
    time_ci_90_lower: float = 999.0,
    time_ci_90_upper: float = 1001.0,
    distance_ci_90_lower_ft: float = 0.0,
    distance_ci_90_upper_ft: float = 5000.0,
    confidence: str = "normal",
) -> TouchdownResult:
    return TouchdownResult(
        flight_id=flight_id,
        aircraft_type=aircraft_type,
        ads_b_source=ads_b_source,
        touchdown_time=touchdown_time,
        along_runway_distance_ft=along_runway_distance_ft,
        lateral_offset_ft=0.0,
        groundspeed_at_touchdown_kt=groundspeed_at_touchdown_kt,
        time_ci_90_lower=time_ci_90_lower,
        time_ci_90_upper=time_ci_90_upper,
        distance_ci_90_lower_ft=distance_ci_90_lower_ft,
        distance_ci_90_upper_ft=distance_ci_90_upper_ft,
        speed_ci_90_lower_kt=groundspeed_at_touchdown_kt - 3.0,
        speed_ci_90_upper_kt=groundspeed_at_touchdown_kt + 3.0,
        trajectory_type="completed-landing",
        confidence=confidence,
        reason_code=None,
        contributing_estimators=["decel_knee"],
        excluded_estimators=[],
        physics_anchor_t_td=touchdown_time,
        physics_anchor_diagnostics=None,
        lever_arm_used=_lever_arm(),
        lever_arm_missing=False,
        assumed_touchdown_pitch_deg=5.5,
        geometric_altitude_available=True,
        runway_elevation_datum="HAE",
        suspected_wrong_runway=False,
        out_of_bounds=False,
        data_version="v",
        code_commit="c",
        config_hash="h",
        model_artifact_hash=None,
    )


def _truth(
    runway: RunwayReference,
    *,
    flight_id: str = "F1",
    along_m: float = 400.0,
    touchdown_time_qar: float = 1000.0,
    clock_offset_quality: str = "good",
    aircraft_type: str = "B738",
    airport_id: str = "KJFK",
    runway_id: str = "04L",
    tail_number: str = "N1",
) -> QARTruthRecord:
    lat, lon = _point_along(runway, along_m)
    return QARTruthRecord(
        flight_id=flight_id,
        touchdown_time_qar=touchdown_time_qar,
        touchdown_lat=lat,
        touchdown_lon=lon,
        clock_offset_estimate=0.0,
        clock_offset_quality=clock_offset_quality,
        aircraft_type=aircraft_type,
        runway_id=runway_id,
        airport_id=airport_id,
        tail_number=tail_number,
    )


def _config(
    *,
    coverage_min: float = 0.85,
    coverage_max: float = 0.95,
    below_target_min_flights: int = 200,
    provisional_targets: ProvisionalAccuracyTargets | None = None,
) -> ValidationConfig:
    if provisional_targets is None:
        provisional_targets = ProvisionalAccuracyTargets(
            distance_rmse_ft=250.0,
            distance_p95_abs_error_ft=400.0,
            distance_p95_long_side_ft=500.0,
            median_signed_error_abs_ft=75.0,
            baseline_improvement_pct=30.0,
        )
    return ValidationConfig(
        primary_split_key="tail",
        generalization_evals=["airport", "runway"],
        use_calibration_split=True,
        train_fraction=0.70,
        calibration_fraction=0.15,
        test_fraction=0.15,
        min_stratum_size=30,
        cross_source=True,
        clock_offset_max_s=2.0,
        clock_drift_max_s=1.0,
        clock_xcorr_resample_dt_s=0.1,
        clock_max_lag_search_s=10.0,
        clock_min_overlap_s=20.0,
        clock_min_peak_correlation=0.5,
        clock_drift_segments=3,
        wrong_runway_lateral_margin_ft=50.0,
        approach_speed_band_edges_kt=(120.0, 140.0, 160.0),
        coverage_min=coverage_min,
        coverage_max=coverage_max,
        below_target_min_flights=below_target_min_flights,
        provisional_targets=provisional_targets,
    )


def _eval(
    runway: RunwayReference,
    *,
    signed_error_ft: float = 0.0,
    truth_along_m: float = 400.0,
    **kwargs,
) -> FlightEvaluation:
    """FlightEvaluation whose system signed distance error is ``signed_error_ft``."""
    truth_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in {"flight_id", "touchdown_time_qar", "clock_offset_quality",
                 "aircraft_type", "airport_id", "runway_id", "tail_number"}
    }
    truth = _truth(runway, along_m=truth_along_m, **truth_kwargs)
    truth_ft = along_runway_truth_distance_ft(
        runway, truth.touchdown_lat, truth.touchdown_lon
    )
    result_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k in {"flight_id", "aircraft_type", "ads_b_source", "touchdown_time",
                 "groundspeed_at_touchdown_kt", "time_ci_90_lower", "time_ci_90_upper",
                 "distance_ci_90_lower_ft", "distance_ci_90_upper_ft", "confidence"}
    }
    result = _result(
        along_runway_distance_ft=truth_ft + signed_error_ft, **result_kwargs
    )
    return FlightEvaluation(result=result, truth=truth, runway=runway)


# ---------------------------------------------------------------------------
# classify_coverage (Req 4.3, 4.4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_classify_coverage_bands():
    """Coverage inside 85-95% is in-band; below is under; above is over.

    Validates: Requirements 4.3, 4.4
    """
    assert classify_coverage(0.90, 0.85, 0.95) == COVERAGE_IN_BAND
    assert classify_coverage(0.85, 0.85, 0.95) == COVERAGE_IN_BAND  # inclusive edge
    assert classify_coverage(0.95, 0.85, 0.95) == COVERAGE_IN_BAND  # inclusive edge
    assert classify_coverage(0.80, 0.85, 0.95) == COVERAGE_UNDER
    assert classify_coverage(0.99, 0.85, 0.95) == COVERAGE_OVER
    assert classify_coverage(float("nan"), 0.85, 0.95) == COVERAGE_UNDEFINED


# ---------------------------------------------------------------------------
# assess_coverage (Req 4.3, 4.4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assess_coverage_distance_and_time_in_band():
    """Both distance and time coverage are measured and classified in-band.

    Validates: Requirements 4.3, 4.4
    """
    runway = _runway()
    config = _config()
    evals = []
    # 20 flights: 18 inside both CIs, 2 outside both -> coverage 0.90 (in-band).
    for i in range(20):
        inside = i >= 2
        if inside:
            evals.append(_eval(
                runway, signed_error_ft=50.0, flight_id=f"F{i}",
                touchdown_time=1000.0, touchdown_time_qar=1000.0,
                time_ci_90_lower=999.0, time_ci_90_upper=1001.0,
                distance_ci_90_lower_ft=0.0, distance_ci_90_upper_ft=5000.0,
            ))
        else:
            # Distance truth well outside the narrow distance CI, QAR time
            # well outside the narrow time CI.
            evals.append(_eval(
                runway, signed_error_ft=900.0, flight_id=f"F{i}",
                touchdown_time=1000.0, touchdown_time_qar=1010.0,
                time_ci_90_lower=999.0, time_ci_90_upper=1001.0,
                distance_ci_90_lower_ft=0.0, distance_ci_90_upper_ft=10.0,
            ))
    assessment = assess_coverage(evals, config)
    assert assessment.n_distance == 20
    assert assessment.n_time == 20
    assert assessment.distance_coverage == pytest.approx(0.90, abs=1e-9)
    assert assessment.time_coverage == pytest.approx(0.90, abs=1e-9)
    assert assessment.distance_classification == COVERAGE_IN_BAND
    assert assessment.time_classification == COVERAGE_IN_BAND


@pytest.mark.unit
def test_assess_coverage_under_and_over():
    """Narrow CIs undercover (unsafe); very wide CIs overcover (uninformative).

    Validates: Requirements 4.3, 4.4
    """
    runway = _runway()
    config = _config()
    # Narrow distance CI never contains truth -> undercovered.
    narrow = [
        _eval(runway, signed_error_ft=500.0, flight_id=f"N{i}",
              distance_ci_90_lower_ft=0.0, distance_ci_90_upper_ft=10.0)
        for i in range(5)
    ]
    assert assess_coverage(narrow, config).distance_classification == COVERAGE_UNDER

    # Extremely wide CI always contains truth -> overcovered (100% > 95%).
    wide = [
        _eval(runway, signed_error_ft=10.0, flight_id=f"W{i}",
              distance_ci_90_lower_ft=-1e6, distance_ci_90_upper_ft=1e6)
        for i in range(5)
    ]
    assert assess_coverage(wide, config).distance_classification == COVERAGE_OVER


@pytest.mark.unit
def test_assess_coverage_time_excludes_failed_clock():
    """Failed-clock flights drop out of the time coverage sample only.

    Validates: Requirements 4.3, 12.10
    """
    runway = _runway()
    config = _config()
    good = _eval(
        runway, signed_error_ft=0.0, flight_id="G", clock_offset_quality="good",
        touchdown_time=1000.0, touchdown_time_qar=1000.0,
        time_ci_90_lower=999.0, time_ci_90_upper=1001.0,
    )
    # Failed-clock flight: its (bogus) QAR time is far outside the time CI, but
    # it must not enter the time coverage sample; it still counts for distance.
    failed = _eval(
        runway, signed_error_ft=0.0, flight_id="B", clock_offset_quality="failed",
        touchdown_time=1000.0, touchdown_time_qar=9000.0,
        time_ci_90_lower=999.0, time_ci_90_upper=1001.0,
    )
    assessment = assess_coverage([good, failed], config)
    assert assessment.n_distance == 2
    assert assessment.n_time == 1  # only the good-clock flight
    assert assessment.time_coverage == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_assess_coverage_empty_is_undefined():
    """With no flights, coverage is undefined rather than an error.

    Validates: Requirements 4.3, 4.4
    """
    assessment = assess_coverage([], _config())
    assert assessment.n_distance == 0
    assert assessment.distance_classification == COVERAGE_UNDEFINED
    assert assessment.time_classification == COVERAGE_UNDEFINED
    assert math.isnan(assessment.distance_coverage)


# ---------------------------------------------------------------------------
# Cadence-limited error floor (Req 13.0, 13.1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cadence_floor_arithmetic():
    """Floor is speed x cadence, converted knots->ft/s (Req 13.0).

    Validates: Requirements 13.0, 13.1
    """
    # 140 kt over a 5 s update interval.
    floor = cadence_limited_floor_ft(140.0, 5.0)
    assert floor == pytest.approx(140.0 * KNOTS_TO_FT_PER_S * 5.0, rel=1e-12)
    # Sanity: ~1181 ft (140 kt ~ 236 ft/s * 5 s).
    assert floor == pytest.approx(1181.5, abs=1.0)
    # Doubling either factor doubles the floor.
    assert cadence_limited_floor_ft(140.0, 10.0) == pytest.approx(2.0 * floor, rel=1e-12)
    assert cadence_limited_floor_ft(280.0, 5.0) == pytest.approx(2.0 * floor, rel=1e-12)


@pytest.mark.unit
def test_cadence_floor_nonfinite():
    """Non-finite inputs yield a NaN floor (no exception).

    Validates: Requirements 13.0
    """
    assert math.isnan(cadence_limited_floor_ft(float("nan"), 5.0))
    assert math.isnan(cadence_limited_floor_ft(140.0, float("inf")))


@pytest.mark.unit
def test_characterize_error_floor_representative_and_fraction():
    """Floor uses the representative speed; fraction_within_floor is per-flight.

    Validates: Requirements 13.0, 13.1
    """
    runway = _runway()
    config = _config()
    cadence_s = 5.0
    # Two flights at 140 kt: per-flight floor ~1181 ft. One flight has abs error
    # 100 ft (< floor -> within), the other 2000 ft (> floor -> not within).
    evals = [
        _eval(runway, signed_error_ft=100.0, flight_id="A",
              groundspeed_at_touchdown_kt=140.0),
        _eval(runway, signed_error_ft=2000.0, flight_id="B",
              groundspeed_at_touchdown_kt=140.0),
    ]
    report = characterize_error_floor(evals, config, cadence_s)
    assert report.n_flights == 2
    assert report.representative_groundspeed_kt == pytest.approx(140.0)
    assert report.floor_ft == pytest.approx(140.0 * KNOTS_TO_FT_PER_S * 5.0, rel=1e-9)
    assert report.per_flight_floor_median_ft == pytest.approx(report.floor_ft, rel=1e-9)
    # One of the two flights is within its floor.
    assert report.fraction_within_floor == pytest.approx(0.5, abs=1e-9)

    # An explicit representative speed overrides the per-flight median.
    report2 = characterize_error_floor(
        evals, config, cadence_s, representative_groundspeed_kt=120.0
    )
    assert report2.floor_ft == pytest.approx(120.0 * KNOTS_TO_FT_PER_S * 5.0, rel=1e-9)


# ---------------------------------------------------------------------------
# Below-target flagging (Req 13.2-13.5)
# ---------------------------------------------------------------------------


def _metrics(
    *,
    n_flights: int,
    rmse: float = 100.0,
    p95_abs: float = 200.0,
    p95_long: float = 250.0,
    median_signed: float = 0.0,
    stratum_key: str | None = None,
) -> ValidationMetrics:
    return ValidationMetrics(
        n_flights=n_flights,
        distance_rmse_ft=rmse,
        distance_median_abs_error_ft=50.0,
        distance_iqr_ft=(20.0, 80.0),
        distance_p95_abs_error_ft=p95_abs,
        distance_p99_abs_error_ft=p95_abs + 50.0,
        distance_p95_long_side_ft=p95_long,
        distance_median_signed_error_ft=median_signed,
        time_rmse_s=1.0,
        time_median_abs_error_s=0.5,
        baseline_rmse_ft=500.0,
        improvement_pct=50.0,
        ci_90_coverage=0.9,
        stratum_key=stratum_key,
    )


def _stratum(dimension: str, key: str, n_flights: int, **metric_kwargs) -> StratumResult:
    return StratumResult(
        dimension=dimension,
        key=key,
        n_flights=n_flights,
        reportable=n_flights >= 30,
        metrics=_metrics(n_flights=n_flights, stratum_key=key, **metric_kwargs),
    )


def _report(strata, below_threshold=()) -> StratifiedMetricsReport:
    return StratifiedMetricsReport(
        overall=_metrics(n_flights=sum(s.n_flights for s in strata) or 1),
        strata=tuple(strata),
        below_threshold=tuple(below_threshold),
        min_stratum_size=30,
        approach_speed_band_edges_kt=(120.0, 140.0, 160.0),
    )


@pytest.mark.unit
def test_flag_below_target_large_stratum_flagged_no_raise():
    """A >=200-flight stratum missing targets is flagged without raising.

    Validates: Requirements 13.1, 13.3, 13.5
    """
    config = _config(below_target_min_flights=200)
    # 250-flight aireon stratum: RMSE 300 > 250 and p95 abs 450 > 400.
    strata = [
        _stratum("source", "aireon", 250, rmse=300.0, p95_abs=450.0),
    ]
    flags = flag_below_target(_report(strata), config)
    assert isinstance(flags, tuple)
    metrics_flagged = {f.metric for f in flags}
    assert "distance_rmse_ft" in metrics_flagged
    assert "distance_p95_abs_error_ft" in metrics_flagged
    for f in flags:
        assert isinstance(f, BelowTargetFlag)
        assert f.direction == "max"
        assert f.observed > f.target


@pytest.mark.unit
def test_flag_below_target_bias_uses_absolute_median_signed():
    """The bias check flags |median signed error| above the target (Req 13.2).

    Validates: Requirements 13.2, 13.5
    """
    config = _config(below_target_min_flights=200)
    # median signed -100 ft -> |bias| 100 > 75 target.
    strata = [_stratum("aircraft_type", "B738", 300, median_signed=-100.0)]
    flags = flag_below_target(_report(strata), config)
    bias = [f for f in flags if f.metric == "distance_median_signed_error_abs_ft"]
    assert len(bias) == 1
    assert bias[0].observed == pytest.approx(100.0)
    assert bias[0].target == pytest.approx(75.0)


@pytest.mark.unit
def test_flag_below_target_small_stratum_not_flagged():
    """Strata below the 200-flight gate are never flagged, even if far off target.

    Validates: Requirements 13.5
    """
    config = _config(below_target_min_flights=200)
    # 150 flights (>=30 reporting gate, but < 200 flagging gate) with awful metrics.
    strata = [_stratum("source", "aireon", 150, rmse=9999.0, p95_abs=9999.0)]
    flags = flag_below_target(_report(strata), config)
    assert flags == ()


@pytest.mark.unit
def test_flag_below_target_within_target_produces_no_flags():
    """A large stratum meeting every target produces no flags.

    Validates: Requirements 13.1, 13.2, 13.3, 13.5
    """
    config = _config(below_target_min_flights=200)
    strata = [
        _stratum("source", "aireon", 500, rmse=200.0, p95_abs=350.0,
                 p95_long=400.0, median_signed=10.0),
    ]
    assert flag_below_target(_report(strata), config) == ()


@pytest.mark.unit
def test_flag_below_target_only_source_and_aircraft_type_dimensions():
    """Only ADS-B source and aircraft-type strata are inspected (Req 13.5).

    Validates: Requirements 13.5
    """
    config = _config(below_target_min_flights=200)
    strata = [
        # Airport/runway/speed-band strata are out of scope even if far off target.
        _stratum("airport", "KJFK", 400, rmse=9999.0),
        _stratum("approach_speed_band", "120-140", 400, rmse=9999.0),
    ]
    assert flag_below_target(_report(strata), config) == ()


@pytest.mark.unit
def test_flag_below_target_nan_long_side_skipped():
    """A NaN long-side percentile is treated as no-data, not a failure.

    Validates: Requirements 13.3, 13.5
    """
    config = _config(below_target_min_flights=200)
    strata = [_stratum("source", "aireon", 300, p95_long=float("nan"))]
    flags = flag_below_target(_report(strata), config)
    assert all(f.metric != "distance_p95_long_side_ft" for f in flags)
