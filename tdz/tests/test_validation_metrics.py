"""Tests for the stratified distance/time validation metrics (Task 22.2).

Covers the clock-independent along-runway distance truth (Req 12.10), the error
metrics (signed/absolute distance error, RMSE, median, IQR, p95/p99 absolute,
p95 long-side signed, time error; Req 12.1, 12.5, 12.6), stratification with the
minimum-stratum-size gate (Req 12.7, 12.9), and the side-by-side naive-baseline
comparison (Req 12.8), plus:

* a known-answer test for the geometric distance truth (projection onto the
  centerline), and its independence from the QAR clock fields;
* arithmetic checks for each metric (signed vs absolute, percentiles, the
  positive-only long-side tail);
* a property test asserting the reported metrics equal a direct NumPy
  computation of the same statistics over arbitrary error distributions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pyproj import Geod

from tdz.config.schema import ValidationConfig
from tdz.models import LeverArm, QARTruthRecord, RunwayReference, TouchdownResult
from tdz.uncertainty import M_TO_FT
from tdz.validation import (
    FlightEvaluation,
    along_runway_truth_distance_ft,
    approach_speed_band_label,
    compute_metrics,
    compute_stratified_metrics,
)

_GEOD = Geod(ellps="WGS84")


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _runway(heading_deg: float = 90.0) -> RunwayReference:
    """A valid runway; heading points due east by default."""
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
    """Return a (lat, lon) exactly ``along_m`` down the centerline from threshold."""
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
    distance_ci_90_lower_ft: float = 0.0,
    distance_ci_90_upper_ft: float = 5000.0,
    confidence: str = "normal",
) -> TouchdownResult:
    """Build a TouchdownResult; only the metric-relevant fields matter here."""
    return TouchdownResult(
        flight_id=flight_id,
        aircraft_type=aircraft_type,
        ads_b_source=ads_b_source,
        touchdown_time=touchdown_time,
        along_runway_distance_ft=along_runway_distance_ft,
        lateral_offset_ft=0.0,
        groundspeed_at_touchdown_kt=groundspeed_at_touchdown_kt,
        time_ci_90_lower=touchdown_time - 1.0,
        time_ci_90_upper=touchdown_time + 1.0,
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


def _validation_config(
    *,
    min_stratum_size: int = 30,
    approach_speed_band_edges_kt: tuple[float, ...] = (120.0, 140.0, 160.0),
) -> ValidationConfig:
    return ValidationConfig(
        primary_split_key="tail",
        generalization_evals=["airport", "runway"],
        use_calibration_split=True,
        train_fraction=0.70,
        calibration_fraction=0.15,
        test_fraction=0.15,
        min_stratum_size=min_stratum_size,
        cross_source=True,
        clock_offset_max_s=2.0,
        clock_drift_max_s=1.0,
        clock_xcorr_resample_dt_s=0.1,
        clock_max_lag_search_s=10.0,
        clock_min_overlap_s=20.0,
        clock_min_peak_correlation=0.5,
        clock_drift_segments=3,
        wrong_runway_lateral_margin_ft=50.0,
        approach_speed_band_edges_kt=approach_speed_band_edges_kt,
    )


def _eval_with_error(
    runway: RunwayReference,
    *,
    signed_error_ft: float,
    truth_along_m: float = 400.0,
    baseline_distance_m: float | None = None,
    **kwargs,
) -> FlightEvaluation:
    """A FlightEvaluation whose system signed distance error is exactly ``signed_error_ft``."""
    truth = _truth(runway, along_m=truth_along_m, **{
        k: v for k, v in kwargs.items()
        if k in {"flight_id", "touchdown_time_qar", "clock_offset_quality",
                 "aircraft_type", "airport_id", "runway_id", "tail_number"}
    })
    truth_ft = along_runway_truth_distance_ft(runway, truth.touchdown_lat, truth.touchdown_lon)
    result = _result(
        along_runway_distance_ft=truth_ft + signed_error_ft,
        **{k: v for k, v in kwargs.items()
           if k in {"flight_id", "aircraft_type", "ads_b_source", "touchdown_time",
                    "groundspeed_at_touchdown_kt", "distance_ci_90_lower_ft",
                    "distance_ci_90_upper_ft", "confidence"}},
    )
    return FlightEvaluation(
        result=result, truth=truth, runway=runway, baseline_distance_m=baseline_distance_m
    )


# ---------------------------------------------------------------------------
# Distance truth (Req 12.10)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_distance_truth_known_answer():
    """Truth distance is the geodesic along-centerline projection, in feet.

    Validates: Requirements 12.10
    """
    runway = _runway()
    lat, lon = _point_along(runway, 762.0)  # 762 m == 2500 ft down the centerline
    truth_ft = along_runway_truth_distance_ft(runway, lat, lon)
    assert truth_ft == pytest.approx(762.0 * M_TO_FT, abs=1e-3)
    assert truth_ft == pytest.approx(2500.0, abs=1e-2)


@pytest.mark.unit
def test_distance_truth_independent_of_clock_fields():
    """The distance truth uses lat/long only; QAR clock fields never enter it.

    Validates: Requirements 12.10
    """
    runway = _runway()
    good = _truth(runway, along_m=500.0, touchdown_time_qar=1000.0, clock_offset_quality="good")
    failed = _truth(runway, along_m=500.0, touchdown_time_qar=9999.0, clock_offset_quality="failed")
    d_good = along_runway_truth_distance_ft(runway, good.touchdown_lat, good.touchdown_lon)
    d_failed = along_runway_truth_distance_ft(runway, failed.touchdown_lat, failed.touchdown_lon)
    assert d_good == pytest.approx(d_failed)


# ---------------------------------------------------------------------------
# Metric arithmetic (Req 12.1, 12.5, 12.6)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_signed_error_sign_convention_positive_is_longer():
    """Positive signed error means the estimate is longer than truth (Req 12.5).

    Validates: Requirements 12.5
    """
    runway = _runway()
    longer = _eval_with_error(runway, signed_error_ft=200.0)
    shorter = _eval_with_error(runway, signed_error_ft=-200.0)
    assert compute_metrics([longer]).distance_median_signed_error_ft == pytest.approx(200.0, abs=1e-3)
    assert compute_metrics([shorter]).distance_median_signed_error_ft == pytest.approx(-200.0, abs=1e-3)


@pytest.mark.unit
def test_distance_metric_arithmetic():
    """RMSE / median / IQR / p95 / p99 match direct computation on known errors.

    Validates: Requirements 12.1, 12.6
    """
    runway = _runway()
    errors = [-300.0, -100.0, 50.0, 150.0, 400.0]
    evals = [_eval_with_error(runway, signed_error_ft=e, flight_id=f"F{i}")
             for i, e in enumerate(errors)]
    m = compute_metrics(evals)

    err = np.array(errors)
    abs_err = np.abs(err)
    assert m.n_flights == 5
    assert m.distance_rmse_ft == pytest.approx(float(np.sqrt(np.mean(err**2))), abs=1e-3)
    assert m.distance_median_abs_error_ft == pytest.approx(float(np.median(abs_err)), abs=1e-3)
    assert m.distance_median_signed_error_ft == pytest.approx(float(np.median(err)), abs=1e-3)
    assert m.distance_iqr_ft[0] == pytest.approx(float(np.percentile(abs_err, 25)), abs=1e-3)
    assert m.distance_iqr_ft[1] == pytest.approx(float(np.percentile(abs_err, 75)), abs=1e-3)
    assert m.distance_p95_abs_error_ft == pytest.approx(float(np.percentile(abs_err, 95)), abs=1e-3)
    assert m.distance_p99_abs_error_ft == pytest.approx(float(np.percentile(abs_err, 99)), abs=1e-3)


@pytest.mark.unit
def test_p95_long_side_uses_positive_errors_only():
    """The long-side percentile is taken over the positive (long) tail only.

    Validates: Requirements 12.6
    """
    runway = _runway()
    errors = [-500.0, -400.0, 100.0, 200.0, 300.0]
    evals = [_eval_with_error(runway, signed_error_ft=e, flight_id=f"F{i}")
             for i, e in enumerate(errors)]
    m = compute_metrics(evals)
    positives = np.array([100.0, 200.0, 300.0])
    assert m.distance_p95_long_side_ft == pytest.approx(float(np.percentile(positives, 95)), abs=1e-3)


@pytest.mark.unit
def test_p95_long_side_nan_when_no_positive_errors():
    """With no long-side errors the positive-tail percentile is NaN.

    Validates: Requirements 12.6
    """
    runway = _runway()
    evals = [_eval_with_error(runway, signed_error_ft=-100.0, flight_id="A"),
             _eval_with_error(runway, signed_error_ft=-50.0, flight_id="B")]
    assert math.isnan(compute_metrics(evals).distance_p95_long_side_ft)


# ---------------------------------------------------------------------------
# Time error + clock handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_time_error_excludes_failed_clock_but_keeps_distance():
    """Failed-clock flights drop out of the time metric yet stay in distance.

    Validates: Requirements 12.10
    """
    runway = _runway()
    # Two good-clock flights with +2s / -2s time error; one failed-clock flight.
    good1 = _eval_with_error(
        runway, signed_error_ft=0.0, flight_id="G1",
        touchdown_time=1002.0, touchdown_time_qar=1000.0, clock_offset_quality="good",
    )
    good2 = _eval_with_error(
        runway, signed_error_ft=0.0, flight_id="G2",
        touchdown_time=998.0, touchdown_time_qar=1000.0, clock_offset_quality="good",
    )
    failed = _eval_with_error(
        runway, signed_error_ft=0.0, flight_id="Bad",
        touchdown_time=5000.0, touchdown_time_qar=1000.0, clock_offset_quality="failed",
    )
    m = compute_metrics([good1, good2, failed])
    # All three counted for distance.
    assert m.n_flights == 3
    # Time RMSE over only the two good flights: sqrt(mean(2^2, 2^2)) == 2.
    assert m.time_rmse_s == pytest.approx(2.0, abs=1e-6)
    assert m.time_median_abs_error_s == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Baseline comparison (Req 12.8)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_baseline_side_by_side_and_improvement():
    """Baseline RMSE and improvement% are reported side-by-side with the system.

    Validates: Requirements 12.8
    """
    runway = _runway()
    truth_along_m = 400.0
    truth_ft = along_runway_truth_distance_ft(runway, *_point_along(runway, truth_along_m))
    # System error +/-100 ft; baseline error +/-500 ft (in meters for the input).
    system_errors = [100.0, -100.0]
    baseline_errors_ft = [500.0, -500.0]
    evals = []
    for i, (se, be) in enumerate(zip(system_errors, baseline_errors_ft)):
        baseline_m = (truth_ft + be) / M_TO_FT
        evals.append(_eval_with_error(
            runway, signed_error_ft=se, truth_along_m=truth_along_m,
            flight_id=f"F{i}", baseline_distance_m=baseline_m,
        ))
    m = compute_metrics(evals)
    assert m.distance_rmse_ft == pytest.approx(100.0, abs=1e-2)
    assert m.baseline_rmse_ft == pytest.approx(500.0, abs=1e-2)
    assert m.improvement_pct == pytest.approx((500.0 - 100.0) / 500.0 * 100.0, abs=1e-2)


@pytest.mark.unit
def test_no_estimate_flights_are_excluded():
    """Flights flagged no-estimate contribute to no metric.

    Validates: Requirements 12.1
    """
    runway = _runway()
    real = _eval_with_error(runway, signed_error_ft=100.0, flight_id="R")
    ghost = _eval_with_error(runway, signed_error_ft=9999.0, flight_id="N", confidence="no-estimate")
    m = compute_metrics([real, ghost])
    assert m.n_flights == 1
    assert m.distance_median_signed_error_ft == pytest.approx(100.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ci_90_coverage_counts_truth_inside_interval():
    """Coverage is the fraction of truths inside the reported 90% distance CI.

    Validates: Requirements 4.4, 12.10
    """
    runway = _runway()
    # In-interval flight (wide CI) and out-of-interval flight (narrow CI).
    inside = _eval_with_error(
        runway, signed_error_ft=50.0, flight_id="IN",
        distance_ci_90_lower_ft=0.0, distance_ci_90_upper_ft=5000.0,
    )
    outside = _eval_with_error(
        runway, signed_error_ft=800.0, flight_id="OUT",
        distance_ci_90_lower_ft=0.0, distance_ci_90_upper_ft=10.0,
    )
    m = compute_metrics([inside, outside])
    assert m.ci_90_coverage == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# Approach-speed banding
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_approach_speed_band_label():
    """Band labels follow the strictly-increasing edge boundaries.

    Validates: Requirements 12.7
    """
    edges = (120.0, 140.0, 160.0)
    assert approach_speed_band_label(100.0, edges) == "<120"
    assert approach_speed_band_label(120.0, edges) == "120-140"
    assert approach_speed_band_label(139.9, edges) == "120-140"
    assert approach_speed_band_label(140.0, edges) == "140-160"
    assert approach_speed_band_label(160.0, edges) == ">=160"
    assert approach_speed_band_label(200.0, edges) == ">=160"
    assert approach_speed_band_label(float("nan"), edges) == "unknown"


# ---------------------------------------------------------------------------
# Stratification + min-size gate (Req 12.7, 12.9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stratification_min_size_gate():
    """Strata below min_stratum_size are suppressed; larger ones are reportable.

    Validates: Requirements 12.7, 12.9
    """
    runway = _runway()
    evals = []
    # 3 aireon flights, 1 flightradar24 flight.
    for i in range(3):
        evals.append(_eval_with_error(
            runway, signed_error_ft=50.0, flight_id=f"A{i}", ads_b_source="aireon",
        ))
    evals.append(_eval_with_error(
        runway, signed_error_ft=50.0, flight_id="B0", ads_b_source="flightradar24",
    ))
    config = _validation_config(min_stratum_size=2)
    report = compute_stratified_metrics(evals, config)

    source_reportable = {s.key for s in report.strata if s.dimension == "source"}
    source_suppressed = {s.key for s in report.below_threshold if s.dimension == "source"}
    assert "aireon" in source_reportable
    assert "flightradar24" in source_suppressed
    # Overall spans every included flight.
    assert report.overall.n_flights == 4
    # Suppressed strata are flagged, not silently reliable.
    assert all(not s.reportable for s in report.below_threshold)
    assert all(s.reportable for s in report.strata)


@pytest.mark.unit
def test_stratification_dimensions_and_default_gate():
    """All four axes are produced; a small stratum fails the default 30 gate.

    Validates: Requirements 12.7, 12.9
    """
    runway = _runway()
    evals = [_eval_with_error(runway, signed_error_ft=25.0, flight_id=f"F{i}",
                              aircraft_type="B738", ads_b_source="aireon",
                              airport_id="KJFK", groundspeed_at_touchdown_kt=135.0)
             for i in range(5)]
    config = _validation_config(min_stratum_size=30)
    report = compute_stratified_metrics(evals, config)

    dims = {s.dimension for s in report.below_threshold}
    assert dims == {"aircraft_type", "source", "airport", "approach_speed_band"}
    # 5 < 30 -> nothing reportable.
    assert report.strata == ()
    # Speed band 135 kt falls in the 120-140 band.
    band_keys = {s.key for s in report.below_threshold if s.dimension == "approach_speed_band"}
    assert band_keys == {"120-140"}


# ---------------------------------------------------------------------------
# Property: reported metrics equal a direct NumPy computation
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    errors=st.lists(
        st.floats(min_value=-2000.0, max_value=2000.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=60,
    )
)
def test_metrics_match_numpy_reference(errors):
    """For any error distribution the metrics equal the direct NumPy statistics.

    Validates: Requirements 12.1, 12.5, 12.6
    """
    runway = _runway()
    evals = [_eval_with_error(runway, signed_error_ft=e, flight_id=f"F{i}")
             for i, e in enumerate(errors)]
    m = compute_metrics(evals)

    # Reference errors are the signed errors actually realized in the built
    # evaluations (system_ft - truth_ft), not the intended inputs: reconstructing
    # a distance as truth_ft + e is lossy for e tiny relative to truth_ft, and the
    # metric correctly reflects the representable value.
    truth_ft = along_runway_truth_distance_ft(
        runway, evals[0].truth.touchdown_lat, evals[0].truth.touchdown_lon
    )
    err = np.asarray(
        [ev.result.along_runway_distance_ft - truth_ft for ev in evals], dtype=float
    )
    abs_err = np.abs(err)
    assert m.n_flights == err.size
    assert m.distance_rmse_ft == pytest.approx(float(np.sqrt(np.mean(err**2))), abs=1e-3)
    assert m.distance_median_signed_error_ft == pytest.approx(float(np.median(err)), abs=1e-3)
    assert m.distance_median_abs_error_ft == pytest.approx(float(np.median(abs_err)), abs=1e-3)
    assert m.distance_p95_abs_error_ft == pytest.approx(float(np.percentile(abs_err, 95)), abs=1e-3)
    assert m.distance_p99_abs_error_ft == pytest.approx(float(np.percentile(abs_err, 99)), abs=1e-3)

    positives = err[err > 0.0]
    if positives.size:
        assert m.distance_p95_long_side_ft == pytest.approx(
            float(np.percentile(positives, 95)), abs=1e-3
        )
    else:
        assert math.isnan(m.distance_p95_long_side_ft)
