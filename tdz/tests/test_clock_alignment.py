"""Tests for QAR<->ADS-B clock alignment (Task 21).

Covers the cross-correlation offset estimator (Req 19.1), within-flight drift
detection (Req 19.2), the time-domain-only exclusion / tagging with retention
for clock-independent distance validation (Req 19.3, 19.6, 19.7), the offset
applied to QAR timestamps only (Req 19.4), the corpus offset-distribution
diagnostic (Req 19.5), plus:

* **P16** -- clock-offset exclusion: any flight whose estimated offset exceeds
  the configured threshold is excluded from the time domain and appears in the
  flagged-flights report.
* the synthetic known-offset recovery test: inject a known offset into an
  otherwise-identical kinematic series and recover it via cross-correlation.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tdz.config.schema import ValidationConfig
from tdz.models import FailureReason, QARTruthRecord
from tdz.validation import (
    QUALITY_FAILED,
    QUALITY_GOOD,
    ClockOffsetResult,
    KinematicSeries,
    align_corpus,
    apply_offset_to_qar,
    estimate_clock_offset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validation_config(
    *,
    clock_offset_max_s: float = 2.0,
    clock_drift_max_s: float = 1.0,
    clock_xcorr_resample_dt_s: float = 0.1,
    clock_max_lag_search_s: float = 10.0,
    clock_min_overlap_s: float = 20.0,
    clock_min_peak_correlation: float = 0.5,
    clock_drift_segments: int = 3,
) -> ValidationConfig:
    """A resolved ValidationConfig carrying the clock-alignment knobs."""
    return ValidationConfig(
        primary_split_key="tail",
        generalization_evals=["airport", "runway"],
        use_calibration_split=True,
        train_fraction=0.70,
        calibration_fraction=0.15,
        test_fraction=0.15,
        min_stratum_size=30,
        cross_source=True,
        clock_offset_max_s=clock_offset_max_s,
        clock_drift_max_s=clock_drift_max_s,
        clock_xcorr_resample_dt_s=clock_xcorr_resample_dt_s,
        clock_max_lag_search_s=clock_max_lag_search_s,
        clock_min_overlap_s=clock_min_overlap_s,
        clock_min_peak_correlation=clock_min_peak_correlation,
        clock_drift_segments=clock_drift_segments,
        wrong_runway_lateral_margin_ft=50.0,
    )


def _decel_profile(t: np.ndarray) -> np.ndarray:
    """A smooth groundspeed profile (m/s) with a distinctive deceleration knee.

    High constant approach speed until a knee near t=60 s, then a smooth
    tanh transition down to a low rollout speed. The knee is a strong,
    unambiguous feature so cross-correlation has a sharp peak (unlike a pure
    linear ramp, which is shift-degenerate after mean removal).
    """
    v_app, v_end, knee, tau = 72.0, 8.0, 60.0, 12.0
    return v_end + (v_app - v_end) * 0.5 * (1.0 - np.tanh((t - knee) / tau))


def _sinusoid_profile(t: np.ndarray) -> np.ndarray:
    """A profile with features across the whole window (for drift detection).

    Period (50 s) is well above the max lag search (10 s), so there is a single
    unambiguous correlation peak within the search window in every sub-segment.
    """
    return 40.0 + 15.0 * np.sin(2.0 * np.pi * t / 50.0)


def _series_with_offset(
    profile,
    offset_s: float,
    *,
    t_start: float = 0.0,
    t_end: float = 120.0,
    sample_dt: float = 1.0,
) -> tuple[KinematicSeries, KinematicSeries]:
    """Build (qar_series, adsb_series) sharing a profile, QAR shifted by offset.

    The ADS-B clock is the reference. A physical event at reference time ``te``
    is stamped in QAR at ``te + offset`` (so the estimated ``QAR - ADS-B`` offset
    should recover ``offset_s``). Both series carry the same kinematic values.
    """
    te = np.arange(t_start, t_end + sample_dt * 0.5, sample_dt)
    values = profile(te)
    adsb = KinematicSeries(times=te, values=values)
    qar = KinematicSeries(times=te + offset_s, values=values)
    return qar, adsb


def _qar_record(flight_id: str = "F1", td_time: float = 1000.0) -> QARTruthRecord:
    return QARTruthRecord(
        flight_id=flight_id,
        touchdown_time_qar=td_time,
        touchdown_lat=33.9425,
        touchdown_lon=-118.4081,
        clock_offset_estimate=None,
        clock_offset_quality="",
        aircraft_type="B738",
        runway_id="24R",
        airport_id="KLAX",
        tail_number="N12345",
    )


# ---------------------------------------------------------------------------
# Synthetic known-offset recovery (Req 19.1)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("true_offset", [-5.0, -1.3, 0.0, 0.7, 2.5, 4.4])
def test_synthetic_known_offset_recovery(true_offset):
    """Inject a known offset into a shared kinematic series and recover it."""
    config = _validation_config()
    qar, adsb = _series_with_offset(_decel_profile, true_offset)
    result = estimate_clock_offset("F1", qar, adsb, config)

    assert result.offset_s is not None
    assert result.offset_s == pytest.approx(true_offset, abs=0.25)
    assert result.peak_correlation > 0.9


@pytest.mark.unit
def test_zero_offset_is_good_and_not_excluded():
    config = _validation_config()
    qar, adsb = _series_with_offset(_decel_profile, 0.0)
    result = estimate_clock_offset("F1", qar, adsb, config)

    assert result.quality == QUALITY_GOOD
    assert result.excluded_time_domain is False
    assert result.reason_code is None


# ---------------------------------------------------------------------------
# Exclusion / tagging (Req 19.6) and retention for distance (Req 19.3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_offset_over_threshold_excluded_and_tagged():
    config = _validation_config(clock_offset_max_s=2.0)
    qar, adsb = _series_with_offset(_decel_profile, 4.0)
    result = estimate_clock_offset("F1", qar, adsb, config)

    assert result.offset_s == pytest.approx(4.0, abs=0.25)
    assert result.excluded_time_domain is True
    assert result.reason_code == FailureReason.CLOCK_OFFSET_EXCEEDED.value
    # Still has a usable offset estimate -> retained for distance validation.
    assert result.offset_s is not None


@pytest.mark.unit
def test_offset_under_threshold_retained():
    config = _validation_config(clock_offset_max_s=2.0)
    qar, adsb = _series_with_offset(_decel_profile, 1.0)
    result = estimate_clock_offset("F1", qar, adsb, config)

    assert result.offset_s == pytest.approx(1.0, abs=0.25)
    assert result.excluded_time_domain is False
    assert result.reason_code is None


# ---------------------------------------------------------------------------
# Estimation failure -> excluded (Req 19.7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_insufficient_overlap_fails_and_excludes():
    config = _validation_config(clock_min_overlap_s=20.0)
    # Series barely overlap (< 20 s common support).
    adsb = KinematicSeries(times=np.arange(0.0, 100.0, 1.0), values=_decel_profile(np.arange(0.0, 100.0, 1.0)))
    late = np.arange(95.0, 200.0, 1.0)
    qar = KinematicSeries(times=late, values=_decel_profile(late))
    result = estimate_clock_offset("F1", qar, adsb, config)

    assert result.quality == QUALITY_FAILED
    assert result.offset_s is None
    assert result.excluded_time_domain is True
    assert result.reason_code == FailureReason.CLOCK_OFFSET_EXCEEDED.value


@pytest.mark.unit
def test_flat_series_weak_correlation_fails():
    config = _validation_config()
    t = np.arange(0.0, 120.0, 1.0)
    flat = np.full_like(t, 50.0)
    result = estimate_clock_offset(
        "F1", KinematicSeries(times=t, values=flat), KinematicSeries(times=t, values=flat), config
    )
    assert result.quality == QUALITY_FAILED
    assert result.excluded_time_domain is True


# ---------------------------------------------------------------------------
# Within-flight drift detection (Req 19.2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_within_flight_drift_detected_and_flagged():
    config = _validation_config(clock_drift_max_s=1.0, clock_offset_max_s=10.0)
    # A drifting QAR clock: the lag grows linearly across the flight from 0 to 3 s.
    te = np.arange(0.0, 150.0, 1.0)
    values = _sinusoid_profile(te)
    adsb = KinematicSeries(times=te, values=values)
    drift = np.linspace(0.0, 3.0, te.size)
    qar = KinematicSeries(times=te + drift, values=values)

    result = estimate_clock_offset("F1", qar, adsb, config)

    assert result.drift_s is not None
    assert result.drift_s > config.clock_drift_max_s
    assert result.drift_exceeded is True
    assert result.excluded_time_domain is True
    assert result.reason_code == FailureReason.CLOCK_OFFSET_EXCEEDED.value


@pytest.mark.unit
def test_constant_offset_no_drift():
    config = _validation_config(clock_drift_max_s=1.0, clock_offset_max_s=10.0)
    qar, adsb = _series_with_offset(_sinusoid_profile, 2.5, t_end=150.0)
    result = estimate_clock_offset("F1", qar, adsb, config)

    assert result.drift_exceeded is False


# ---------------------------------------------------------------------------
# Offset applied to QAR timestamps only (Req 19.4, 19.3)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_apply_offset_shifts_time_not_position():
    config = _validation_config()
    qar_series, adsb = _series_with_offset(_decel_profile, 1.5)
    result = estimate_clock_offset("F1", qar_series, adsb, config)
    record = _qar_record(td_time=1000.0)

    aligned = apply_offset_to_qar(record, result)

    # Time shifted by -offset (QAR - ADS-B); lat/long untouched (Req 19.3).
    assert aligned.touchdown_time_qar == pytest.approx(1000.0 - result.offset_s)
    assert aligned.touchdown_lat == record.touchdown_lat
    assert aligned.touchdown_lon == record.touchdown_lon
    assert aligned.clock_offset_estimate == pytest.approx(result.offset_s)
    assert aligned.clock_offset_quality == result.quality


@pytest.mark.unit
def test_apply_offset_failed_leaves_time_unchanged():
    failed = ClockOffsetResult(
        flight_id="F1",
        offset_s=None,
        quality=QUALITY_FAILED,
        drift_s=None,
        drift_exceeded=False,
        overlap_s=0.0,
        peak_correlation=0.0,
        excluded_time_domain=True,
        reason_code=FailureReason.CLOCK_OFFSET_EXCEEDED.value,
    )
    record = _qar_record(td_time=1000.0)
    aligned = apply_offset_to_qar(record, failed)

    assert aligned.touchdown_time_qar == 1000.0
    assert aligned.clock_offset_estimate is None
    assert aligned.clock_offset_quality == QUALITY_FAILED


# ---------------------------------------------------------------------------
# Corpus report: distribution diagnostic (Req 19.5) + flagged report (Req 19.6)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_align_corpus_distribution_and_flagged_report():
    config = _validation_config(clock_offset_max_s=2.0)
    series = {
        "small_a": _series_with_offset(_decel_profile, 0.5),
        "small_b": _series_with_offset(_decel_profile, -0.5),
        "big": _series_with_offset(_decel_profile, 5.0),
    }
    report = align_corpus(series, config)

    # Distribution over reliably-estimated offsets (all three succeeded here).
    assert report.distribution.n == 3
    assert report.distribution.median_s == pytest.approx(0.5, abs=0.25)
    # p95 of |offsets| {0.5, 0.5, 5.0} lands high (near the large offset).
    assert report.distribution.p95_abs_s >= 4.5

    # The over-threshold flight is flagged; the small ones are not.
    flagged_ids = {r.flight_id for r in report.flagged_flights}
    assert "big" in flagged_ids
    assert "small_a" not in flagged_ids
    assert "small_b" not in flagged_ids


# ---------------------------------------------------------------------------
# Property 16: Clock Offset Exclusion
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    true_offset=st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False)
)
def test_p16_clock_offset_exclusion(true_offset):
    """Feature: touchdown-point-detection, Property 16: Clock Offset Exclusion

    For any flight whose estimated QAR-to-ADS-B clock offset exceeds the
    configured threshold (default 2 s), that flight is excluded from time-domain
    training/validation and appears in the flagged-flights report.

    Validates: Requirements 19.4
    """
    threshold = 2.0
    # Avoid the exact decision boundary where sub-grid estimation noise could
    # flip the (estimated-offset > threshold) decision.
    assume(abs(abs(true_offset) - threshold) > 0.3)

    config = _validation_config(clock_offset_max_s=threshold)
    qar, adsb = _series_with_offset(_decel_profile, true_offset)
    report = align_corpus({"F1": (qar, adsb)}, config)
    result = report.results[0]

    assert result.offset_s is not None
    # The estimator recovers the injected offset.
    assert result.offset_s == pytest.approx(true_offset, abs=0.3)

    over_threshold = abs(result.offset_s) > threshold
    assert result.excluded_time_domain == over_threshold

    if over_threshold:
        assert result.reason_code == FailureReason.CLOCK_OFFSET_EXCEEDED.value
        assert result in report.flagged_flights
    else:
        assert result not in report.flagged_flights
