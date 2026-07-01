"""Tests for the position gates (Task 7).

Covers Property 22 (Wrong-Runway Lateral-Offset Gate) plus known-answer / unit
tests for the lateral-offset boundary (strictly greater-than trips it), the
sign-via-magnitude handling, the along-runway out-of-bounds gate (Req 2.4), the
feet -> meters margin conversion, and the ValidationConfig-driven margin.

These gates set non-fatal FLAGS (the estimate is still produced and reported);
they never raise, in contrast to the fatal InvalidRunwayReferenceError.
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tdz.config.schema import ValidationConfig
from tdz.geo import (
    FT_TO_M,
    PositionGateResult,
    evaluate_position_gates,
    is_out_of_bounds,
    is_suspected_wrong_runway,
    wrong_runway_lateral_threshold_m,
)
from tdz.geo.projection import ProjectedPosition
from tdz.models import FailureReason, RunwayReference

# Geometry equality is exact arithmetic; allow only floating-point slack.
_TOL_M = 1e-9


def _runway(*, width_m: float = 45.0, length_m: float = 3000.0) -> RunwayReference:
    """Build a valid RunwayReference with the given width/length."""
    return RunwayReference(
        threshold_lat=40.0,
        threshold_lon=-75.0,
        heading_deg=90.0,
        elevation_m=30.0,
        elevation_datum="HAE",
        geoid_undulation_m=0.0,
        length_m=length_m,
        width_m=width_m,
        displaced=False,
    )


def _projected(
    *, along_m: float = 500.0, lateral_m: float = 0.0
) -> ProjectedPosition:
    """Build a ProjectedPosition for gate evaluation."""
    return ProjectedPosition(
        along_runway_distance_m=along_m,
        lateral_offset_m=lateral_m,
    )


def _validation_config(margin_ft: float = 50.0) -> ValidationConfig:
    """Build a ValidationConfig carrying the wrong-runway margin (feet)."""
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
        wrong_runway_lateral_margin_ft=margin_ft,
    )


# ---------------------------------------------------------------------------
# Property 22: Wrong-Runway Lateral-Offset Gate
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    width_m=st.floats(min_value=1.0, max_value=100.0),
    margin_m=st.floats(min_value=0.0, max_value=100.0),
    lateral_offset_m=st.floats(min_value=-600.0, max_value=600.0),
)
def test_wrong_runway_lateral_offset_gate(width_m, margin_m, lateral_offset_m):
    """Feature: touchdown-point-detection, Property 22: Wrong-Runway Lateral-Offset Gate

    For any runway width and margin, a lateral offset whose magnitude exceeds
    half-width + margin sets suspected_wrong_runway=True and includes
    SUSPECTED_WRONG_RUNWAY in the reason codes; an offset within the threshold
    does not. Sign is handled by magnitude (both left/right offsets behave the
    same).
    """
    threshold = width_m / 2.0 + margin_m
    # Avoid the measure-zero exact-boundary case so the iff is unambiguous.
    assume(abs(abs(lateral_offset_m) - threshold) > 1e-6)

    runway = _runway(width_m=width_m)
    projected = _projected(along_m=runway.length_m / 2.0, lateral_m=lateral_offset_m)
    result = evaluate_position_gates(
        projected, runway, wrong_runway_margin_m=margin_m
    )

    expected = abs(lateral_offset_m) > threshold
    assert result.suspected_wrong_runway is expected
    assert (FailureReason.SUSPECTED_WRONG_RUNWAY in result.reason_codes) is expected
    # The reported threshold matches half-width + margin exactly.
    assert result.lateral_threshold_m == pytest.approx(threshold, abs=_TOL_M)
    # The gate is non-fatal: the offset value is still carried through.
    assert result.lateral_offset_m == pytest.approx(lateral_offset_m, abs=_TOL_M)


@pytest.mark.property
@given(
    width_m=st.floats(min_value=1.0, max_value=100.0),
    margin_m=st.floats(min_value=0.0, max_value=100.0),
    excess_m=st.floats(min_value=1.0, max_value=500.0),
    sign=st.sampled_from([-1.0, 1.0]),
)
def test_parallel_runway_swap_always_trips(width_m, margin_m, excess_m, sign):
    """A parallel-runway-swap-style large offset always trips the gate.

    Any offset strictly beyond half-width + margin (either side) is flagged.
    """
    threshold = width_m / 2.0 + margin_m
    lateral = sign * (threshold + excess_m)
    runway = _runway(width_m=width_m)
    result = evaluate_position_gates(
        _projected(lateral_m=lateral), runway, wrong_runway_margin_m=margin_m
    )
    assert result.suspected_wrong_runway is True
    assert FailureReason.SUSPECTED_WRONG_RUNWAY in result.reason_codes


# ---------------------------------------------------------------------------
# Lateral-offset boundary (strictly greater-than) and sign handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_offset_exactly_at_threshold_does_not_flag():
    """Boundary is exclusive: offset exactly at half-width + margin is NOT flagged."""
    width_m, margin_m = 45.0, 15.24
    threshold = width_m / 2.0 + margin_m  # 37.74 m
    runway = _runway(width_m=width_m)
    result = evaluate_position_gates(
        _projected(lateral_m=threshold), runway, wrong_runway_margin_m=margin_m
    )
    assert result.suspected_wrong_runway is False
    assert FailureReason.SUSPECTED_WRONG_RUNWAY not in result.reason_codes
    # Strictly beyond by an epsilon does flag.
    result_over = evaluate_position_gates(
        _projected(lateral_m=threshold + 1e-3), runway, wrong_runway_margin_m=margin_m
    )
    assert result_over.suspected_wrong_runway is True


@pytest.mark.unit
def test_negative_and_positive_offsets_handled_by_magnitude():
    """Left (negative) and right (positive) offsets of equal magnitude match."""
    width_m, margin_m = 45.0, 15.0
    big = width_m / 2.0 + margin_m + 10.0
    runway = _runway(width_m=width_m)
    pos = evaluate_position_gates(
        _projected(lateral_m=big), runway, wrong_runway_margin_m=margin_m
    )
    neg = evaluate_position_gates(
        _projected(lateral_m=-big), runway, wrong_runway_margin_m=margin_m
    )
    assert pos.suspected_wrong_runway is True
    assert neg.suspected_wrong_runway is True


@pytest.mark.unit
def test_is_suspected_wrong_runway_helper():
    """The standalone helper matches the strictly-greater-than rule."""
    assert is_suspected_wrong_runway(40.0, width_m=45.0, margin_m=15.0) is True
    assert is_suspected_wrong_runway(37.5, width_m=45.0, margin_m=15.0) is False
    # Exact boundary (offset == half-width + margin = 22.5) -> not flagged.
    assert is_suspected_wrong_runway(22.5, width_m=45.0, margin_m=0.0) is False


@pytest.mark.unit
def test_wrong_runway_lateral_threshold_helper():
    """Threshold is half-width + margin."""
    assert wrong_runway_lateral_threshold_m(45.0, 15.0) == pytest.approx(37.5, abs=_TOL_M)
    assert wrong_runway_lateral_threshold_m(60.0, 0.0) == pytest.approx(30.0, abs=_TOL_M)


# ---------------------------------------------------------------------------
# Along-runway out-of-bounds gate (Req 2.4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_along_distance_just_past_length_is_out_of_bounds():
    """Distance > runway length -> out_of_bounds flag + code; value retained."""
    runway = _runway(length_m=3000.0)
    result = evaluate_position_gates(
        _projected(along_m=3000.0 + 0.5), runway, wrong_runway_margin_m=15.0
    )
    assert result.out_of_bounds is True
    assert FailureReason.OUT_OF_BOUNDS_POSITION in result.reason_codes
    # The value is still reported for diagnostics (not clamped/dropped).
    assert result.along_runway_distance_m == pytest.approx(3000.5, abs=_TOL_M)


@pytest.mark.unit
def test_negative_along_distance_is_out_of_bounds():
    """Distance < 0 (touchdown before threshold) -> out_of_bounds flag + code."""
    runway = _runway(length_m=3000.0)
    result = evaluate_position_gates(
        _projected(along_m=-0.5), runway, wrong_runway_margin_m=15.0
    )
    assert result.out_of_bounds is True
    assert FailureReason.OUT_OF_BOUNDS_POSITION in result.reason_codes
    assert result.along_runway_distance_m == pytest.approx(-0.5, abs=_TOL_M)


@pytest.mark.unit
def test_along_distance_boundaries_are_inclusive():
    """Exactly 0 and exactly length_m are in-bounds (inclusive [0, length_m])."""
    runway = _runway(length_m=3000.0)
    at_zero = evaluate_position_gates(
        _projected(along_m=0.0), runway, wrong_runway_margin_m=15.0
    )
    at_length = evaluate_position_gates(
        _projected(along_m=3000.0), runway, wrong_runway_margin_m=15.0
    )
    assert at_zero.out_of_bounds is False
    assert at_length.out_of_bounds is False


@pytest.mark.unit
def test_is_out_of_bounds_helper():
    """The standalone helper matches the [0, length_m] inclusive rule."""
    assert is_out_of_bounds(-1.0, length_m=3000.0) is True
    assert is_out_of_bounds(3000.1, length_m=3000.0) is True
    assert is_out_of_bounds(0.0, length_m=3000.0) is False
    assert is_out_of_bounds(3000.0, length_m=3000.0) is False
    assert is_out_of_bounds(1500.0, length_m=3000.0) is False


# ---------------------------------------------------------------------------
# In-bounds normal case
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_in_bounds_normal_case_sets_no_flags():
    """A centered, in-bounds touchdown sets both flags False and no codes."""
    runway = _runway(width_m=45.0, length_m=3000.0)
    result = evaluate_position_gates(
        _projected(along_m=450.0, lateral_m=2.0), runway, wrong_runway_margin_m=15.24
    )
    assert result.suspected_wrong_runway is False
    assert result.out_of_bounds is False
    assert result.reason_codes == ()


@pytest.mark.unit
def test_both_gates_can_trip_together():
    """A long, far-offset touchdown trips both gates; both codes are present."""
    runway = _runway(width_m=45.0, length_m=3000.0)
    result = evaluate_position_gates(
        _projected(along_m=3500.0, lateral_m=500.0),
        runway,
        wrong_runway_margin_m=15.0,
    )
    assert result.suspected_wrong_runway is True
    assert result.out_of_bounds is True
    assert FailureReason.SUSPECTED_WRONG_RUNWAY in result.reason_codes
    assert FailureReason.OUT_OF_BOUNDS_POSITION in result.reason_codes


# ---------------------------------------------------------------------------
# Feet -> meters margin conversion and ValidationConfig-driven margin
# ---------------------------------------------------------------------------


def test_ft_to_m_constant():
    """The documented FT->M conversion factor is exactly 0.3048."""
    assert FT_TO_M == 0.3048


@pytest.mark.unit
def test_50ft_margin_behaves_as_15_24_m():
    """A 50 ft margin via ValidationConfig converts to 15.24 m exactly."""
    runway = _runway(width_m=45.0)  # half-width 22.5 m
    config = _validation_config(margin_ft=50.0)
    result = evaluate_position_gates(_projected(lateral_m=0.0), runway, validation_config=config)
    # half-width (22.5) + 50 ft (15.24 m) = 37.74 m
    assert result.margin_m == pytest.approx(15.24, abs=_TOL_M)
    assert result.lateral_threshold_m == pytest.approx(37.74, abs=_TOL_M)

    # An offset just inside is not flagged; just outside is.
    inside = evaluate_position_gates(
        _projected(lateral_m=37.74 - 1e-3), runway, validation_config=config
    )
    outside = evaluate_position_gates(
        _projected(lateral_m=37.74 + 1e-3), runway, validation_config=config
    )
    assert inside.suspected_wrong_runway is False
    assert outside.suspected_wrong_runway is True


@pytest.mark.unit
def test_validation_config_drives_the_margin():
    """Different config margins move the threshold accordingly."""
    runway = _runway(width_m=45.0)
    tight = _validation_config(margin_ft=0.0)
    wide = _validation_config(margin_ft=100.0)
    tight_result = evaluate_position_gates(
        _projected(lateral_m=25.0), runway, validation_config=tight
    )
    wide_result = evaluate_position_gates(
        _projected(lateral_m=25.0), runway, validation_config=wide
    )
    # 25 m > 22.5 m (tight threshold) -> flagged.
    assert tight_result.suspected_wrong_runway is True
    # 25 m < 22.5 + 30.48 m (wide threshold) -> not flagged.
    assert wide_result.suspected_wrong_runway is False


@pytest.mark.unit
def test_explicit_margin_overrides_config():
    """An explicit meters margin takes precedence over a ValidationConfig."""
    runway = _runway(width_m=45.0)
    config = _validation_config(margin_ft=100.0)  # would give a wide threshold
    result = evaluate_position_gates(
        _projected(lateral_m=25.0),
        runway,
        validation_config=config,
        wrong_runway_margin_m=0.0,  # explicit override -> tight threshold
    )
    assert result.margin_m == pytest.approx(0.0, abs=_TOL_M)
    assert result.suspected_wrong_runway is True


@pytest.mark.unit
def test_missing_margin_raises_value_error():
    """Neither a config nor an explicit margin -> ValueError (no hard-coded margin)."""
    runway = _runway()
    with pytest.raises(ValueError):
        evaluate_position_gates(_projected(), runway)


# ---------------------------------------------------------------------------
# Result object shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_result_is_frozen():
    """PositionGateResult is an immutable value object."""
    runway = _runway()
    result = evaluate_position_gates(
        _projected(), runway, wrong_runway_margin_m=15.0
    )
    assert isinstance(result, PositionGateResult)
    with pytest.raises(Exception):
        result.out_of_bounds = True  # type: ignore[misc]


@pytest.mark.unit
def test_reason_codes_is_tuple():
    """reason_codes is a tuple of FailureReason for the diagnostics record."""
    runway = _runway()
    result = evaluate_position_gates(
        _projected(along_m=-1.0, lateral_m=999.0), runway, wrong_runway_margin_m=15.0
    )
    assert isinstance(result.reason_codes, tuple)
    assert all(isinstance(code, FailureReason) for code in result.reason_codes)
