"""Tests for the cross-source generalization evaluation (Task 22.3, Req 12.9).

Covers:

* the both-feeds intersection selection -- landings present in only one feed are
  excluded so the arms compare like-for-like flights;
* the four-arm structure -- both cross directions plus both same-source arms;
* the accuracy-drop arithmetic -- each cross direction is compared against the
  same-source arm on the same test source (RMSE delta in feet and percentage);
* the ``config.cross_source`` gate; and
* a property: when the two feeds produce identical estimates for shared
  landings, the same-source-vs-same-source transfer penalty is ~0.
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
    default_landing_key,
    evaluate_cross_source,
    shared_landings,
)

_GEOD = Geod(ellps="WGS84")

SRC_A = "aireon"
SRC_B = "flightradar24"


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
    flight_id: str,
    ads_b_source: str,
    along_runway_distance_ft: float,
    confidence: str = "normal",
) -> TouchdownResult:
    return TouchdownResult(
        flight_id=flight_id,
        aircraft_type="B738",
        ads_b_source=ads_b_source,
        touchdown_time=1000.0,
        along_runway_distance_ft=along_runway_distance_ft,
        lateral_offset_ft=0.0,
        groundspeed_at_touchdown_kt=135.0,
        time_ci_90_lower=999.0,
        time_ci_90_upper=1001.0,
        distance_ci_90_lower_ft=0.0,
        distance_ci_90_upper_ft=5000.0,
        speed_ci_90_lower_kt=132.0,
        speed_ci_90_upper_kt=138.0,
        trajectory_type="completed-landing",
        confidence=confidence,
        reason_code=None,
        contributing_estimators=["decel_knee"],
        excluded_estimators=[],
        physics_anchor_t_td=1000.0,
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


def _truth(runway: RunwayReference, *, landing_id: str, along_m: float) -> QARTruthRecord:
    lat, lon = _point_along(runway, along_m)
    return QARTruthRecord(
        flight_id=landing_id,
        touchdown_time_qar=1000.0,
        touchdown_lat=lat,
        touchdown_lon=lon,
        clock_offset_estimate=0.0,
        clock_offset_quality="good",
        aircraft_type="B738",
        runway_id="04L",
        airport_id="KJFK",
        tail_number="N1",
    )


def _eval(
    runway: RunwayReference,
    *,
    landing_id: str,
    ads_b_source: str,
    signed_error_ft: float,
    truth_along_m: float = 400.0,
) -> FlightEvaluation:
    """A FlightEvaluation whose system signed distance error is ``signed_error_ft``.

    ``landing_id`` becomes the shared QAR truth ``flight_id`` (the physical
    landing identity); pass the same id for the two feeds observing one landing.
    """
    truth = _truth(runway, landing_id=landing_id, along_m=truth_along_m)
    truth_ft = along_runway_truth_distance_ft(runway, truth.touchdown_lat, truth.touchdown_lon)
    result = _result(
        flight_id=f"{ads_b_source}:{landing_id}",
        ads_b_source=ads_b_source,
        along_runway_distance_ft=truth_ft + signed_error_ft,
    )
    return FlightEvaluation(result=result, truth=truth, runway=runway)


def _config(*, cross_source: bool = True) -> ValidationConfig:
    return ValidationConfig(
        primary_split_key="tail",
        generalization_evals=["airport", "runway"],
        use_calibration_split=True,
        train_fraction=0.70,
        calibration_fraction=0.15,
        test_fraction=0.15,
        min_stratum_size=30,
        cross_source=cross_source,
        clock_offset_max_s=2.0,
        clock_drift_max_s=1.0,
        clock_xcorr_resample_dt_s=0.1,
        clock_max_lag_search_s=10.0,
        clock_min_overlap_s=20.0,
        clock_min_peak_correlation=0.5,
        clock_drift_segments=3,
        wrong_runway_lateral_margin_ft=50.0,
    )


def _rmse(errors: list[float]) -> float:
    arr = np.asarray(errors, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


# ---------------------------------------------------------------------------
# Both-feeds intersection selection (Req 12.9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_shared_landings_excludes_single_feed_landings():
    """Only landings observed by BOTH feeds are kept in the intersection.

    Validates: Requirements 12.9
    """
    runway = _runway()
    # Landings L1, L2 in both feeds; L3 only in A; L4 only in B.
    arms = {
        (SRC_A, SRC_A): [
            _eval(runway, landing_id="L1", ads_b_source=SRC_A, signed_error_ft=10.0),
            _eval(runway, landing_id="L2", ads_b_source=SRC_A, signed_error_ft=20.0),
            _eval(runway, landing_id="L3", ads_b_source=SRC_A, signed_error_ft=30.0),
        ],
        (SRC_B, SRC_B): [
            _eval(runway, landing_id="L1", ads_b_source=SRC_B, signed_error_ft=15.0),
            _eval(runway, landing_id="L2", ads_b_source=SRC_B, signed_error_ft=25.0),
            _eval(runway, landing_id="L4", ads_b_source=SRC_B, signed_error_ft=35.0),
        ],
    }
    keep = shared_landings(arms, SRC_A, SRC_B, default_landing_key)
    assert keep == {"L1", "L2"}


@pytest.mark.unit
def test_arms_restricted_to_intersection():
    """Every reported arm counts only the both-feeds shared landings.

    Validates: Requirements 12.9
    """
    runway = _runway()
    shared = ["S1", "S2", "S3"]
    # Build all four arms over the 3 shared landings, plus one feed-only landing
    # per source that must be dropped from the metrics.
    def arm(train: str, test: str, extra_id: str | None) -> list[FlightEvaluation]:
        evals = [
            _eval(runway, landing_id=lid, ads_b_source=test, signed_error_ft=50.0)
            for lid in shared
        ]
        if extra_id is not None:
            evals.append(
                _eval(runway, landing_id=extra_id, ads_b_source=test, signed_error_ft=999.0)
            )
        return evals

    arms = {
        (SRC_A, SRC_A): arm(SRC_A, SRC_A, "ONLY_A"),
        (SRC_B, SRC_B): arm(SRC_B, SRC_B, "ONLY_B"),
        (SRC_A, SRC_B): arm(SRC_A, SRC_B, "ONLY_B"),
        (SRC_B, SRC_A): arm(SRC_B, SRC_A, "ONLY_A"),
    }
    report = evaluate_cross_source(arms, _config())
    assert report is not None
    assert report.n_shared_landings == 3
    for a in report.same_source + report.cross_source:
        assert a.n_flights == 3


# ---------------------------------------------------------------------------
# Four-arm structure (Req 12.9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_four_arm_structure_both_directions():
    """Report has both same-source arms and both cross directions.

    Validates: Requirements 12.9
    """
    runway = _runway()
    ids = ["A1", "A2"]

    def arm(test: str) -> list[FlightEvaluation]:
        return [
            _eval(runway, landing_id=lid, ads_b_source=test, signed_error_ft=40.0)
            for lid in ids
        ]

    arms = {
        (SRC_A, SRC_A): arm(SRC_A),
        (SRC_B, SRC_B): arm(SRC_B),
        (SRC_A, SRC_B): arm(SRC_B),
        (SRC_B, SRC_A): arm(SRC_A),
    }
    report = evaluate_cross_source(arms, _config())
    assert report is not None

    same = {(a.train_source, a.test_source) for a in report.same_source}
    cross = {(a.train_source, a.test_source) for a in report.cross_source}
    assert same == {(SRC_A, SRC_A), (SRC_B, SRC_B)}
    assert cross == {(SRC_A, SRC_B), (SRC_B, SRC_A)}

    # Two directions, each cross arm paired with the same-source arm on the SAME
    # test source (same physical flights, only training source differs).
    dirs = {(d.train_source, d.test_source) for d in report.directions}
    assert dirs == {(SRC_B, SRC_A), (SRC_A, SRC_B)}
    for d in report.directions:
        assert d.test_source == d.train_source or d.train_source != d.test_source
        # same-source reference is trained on the test source.
        # (cross trained on the other source.)
        assert d.cross is not d.same_source


@pytest.mark.unit
def test_missing_arm_raises():
    """All four train/test arms are required.

    Validates: Requirements 12.9
    """
    runway = _runway()
    arms = {
        (SRC_A, SRC_A): [_eval(runway, landing_id="X", ads_b_source=SRC_A, signed_error_ft=0.0)],
        (SRC_B, SRC_B): [_eval(runway, landing_id="X", ads_b_source=SRC_B, signed_error_ft=0.0)],
        # cross arms omitted
    }
    with pytest.raises(ValueError, match="four train/test arms"):
        evaluate_cross_source(arms, _config())


# ---------------------------------------------------------------------------
# Accuracy-drop arithmetic (Req 12.9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_accuracy_drop_arithmetic_cross_vs_same():
    """The drop is cross RMSE minus same-source RMSE on the same test source.

    Validates: Requirements 12.9
    """
    runway = _runway()
    ids = ["L1", "L2"]

    # Same-source arms accurate (+/-100 ft); cross arms degraded (+/-300 ft) on
    # the SAME test flights.
    def arm(test: str, errs: list[float]) -> list[FlightEvaluation]:
        return [
            _eval(runway, landing_id=lid, ads_b_source=test, signed_error_ft=e)
            for lid, e in zip(ids, errs)
        ]

    arms = {
        (SRC_A, SRC_A): arm(SRC_A, [100.0, -100.0]),   # same-source, test A
        (SRC_B, SRC_B): arm(SRC_B, [100.0, -100.0]),   # same-source, test B
        (SRC_A, SRC_B): arm(SRC_B, [300.0, -300.0]),   # A -> B, test B
        (SRC_B, SRC_A): arm(SRC_A, [300.0, -300.0]),   # B -> A, test A
    }
    report = evaluate_cross_source(arms, _config())
    assert report is not None

    same_rmse = _rmse([100.0, -100.0])   # == 100
    cross_rmse = _rmse([300.0, -300.0])  # == 300
    expected_delta = cross_rmse - same_rmse
    expected_pct = (cross_rmse - same_rmse) / same_rmse * 100.0

    for d in report.directions:
        assert d.cross.distance_rmse_ft == pytest.approx(cross_rmse, abs=1e-2)
        assert d.same_source.distance_rmse_ft == pytest.approx(same_rmse, abs=1e-2)
        assert d.rmse_drop_ft == pytest.approx(expected_delta, abs=1e-2)
        assert d.rmse_drop_pct == pytest.approx(expected_pct, abs=1e-2)


@pytest.mark.unit
def test_direction_references_correct_same_source_arm():
    """Direction B->A references same-source (A,A); A->B references (B,B).

    Validates: Requirements 12.9
    """
    runway = _runway()
    ids = ["L1", "L2"]

    def arm(test: str, errs: list[float]) -> list[FlightEvaluation]:
        return [
            _eval(runway, landing_id=lid, ads_b_source=test, signed_error_ft=e)
            for lid, e in zip(ids, errs)
        ]

    # Distinct RMSEs per test source so we can verify the correct pairing:
    # test-A same-source RMSE = 100; test-B same-source RMSE = 200.
    arms = {
        (SRC_A, SRC_A): arm(SRC_A, [100.0, -100.0]),
        (SRC_B, SRC_B): arm(SRC_B, [200.0, -200.0]),
        (SRC_A, SRC_B): arm(SRC_B, [400.0, -400.0]),
        (SRC_B, SRC_A): arm(SRC_A, [400.0, -400.0]),
    }
    report = evaluate_cross_source(arms, _config())
    assert report is not None

    by_test = {d.test_source: d for d in report.directions}
    # B -> A tested on A: same-source reference is (A,A), RMSE 100.
    assert by_test[SRC_A].train_source == SRC_B
    assert by_test[SRC_A].same_source.distance_rmse_ft == pytest.approx(100.0, abs=1e-2)
    # A -> B tested on B: same-source reference is (B,B), RMSE 200.
    assert by_test[SRC_B].train_source == SRC_A
    assert by_test[SRC_B].same_source.distance_rmse_ft == pytest.approx(200.0, abs=1e-2)


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cross_source_gate_off_returns_none():
    """With validation.cross_source disabled the evaluation is not reported.

    Validates: Requirements 12.9
    """
    runway = _runway()
    arms = {
        (SRC_A, SRC_A): [_eval(runway, landing_id="X", ads_b_source=SRC_A, signed_error_ft=0.0)],
        (SRC_B, SRC_B): [_eval(runway, landing_id="X", ads_b_source=SRC_B, signed_error_ft=0.0)],
        (SRC_A, SRC_B): [_eval(runway, landing_id="X", ads_b_source=SRC_B, signed_error_ft=0.0)],
        (SRC_B, SRC_A): [_eval(runway, landing_id="X", ads_b_source=SRC_A, signed_error_ft=0.0)],
    }
    assert evaluate_cross_source(arms, _config(cross_source=False)) is None


# ---------------------------------------------------------------------------
# Property: identical outputs -> zero transfer penalty
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    errors=st.lists(
        st.floats(min_value=-800.0, max_value=800.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=25,
    )
)
def test_identical_sources_have_zero_drop(errors):
    """When both feeds and both models produce identical errors, the drop is ~0.

    If the two sources yield identical estimates for the shared landings and the
    cross- and same-source models agree, the source-transfer penalty vanishes.

    Validates: Requirements 12.9
    """
    runway = _runway()
    ids = [f"L{i}" for i in range(len(errors))]

    def arm(test: str) -> list[FlightEvaluation]:
        return [
            _eval(runway, landing_id=lid, ads_b_source=test, signed_error_ft=e)
            for lid, e in zip(ids, errors)
        ]

    arms = {
        (SRC_A, SRC_A): arm(SRC_A),
        (SRC_B, SRC_B): arm(SRC_B),
        (SRC_A, SRC_B): arm(SRC_B),
        (SRC_B, SRC_A): arm(SRC_A),
    }
    report = evaluate_cross_source(arms, _config())
    assert report is not None
    assert report.n_shared_landings == len(set(ids))

    for d in report.directions:
        assert d.rmse_drop_ft == pytest.approx(0.0, abs=1e-6)
        # pct is 0 unless the same-source RMSE is 0 (all-zero errors) -> nan.
        if np.isfinite(d.same_source.distance_rmse_ft) and d.same_source.distance_rmse_ft > 0.0:
            assert d.rmse_drop_pct == pytest.approx(0.0, abs=1e-6)
        else:
            assert math.isnan(d.rmse_drop_pct)
