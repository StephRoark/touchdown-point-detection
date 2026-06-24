"""Tests for the pitch-resolved lever-arm correction (Task 6).

Covers Property 2 (Lever-Arm Correction Geometric Consistency) and Property 23
(Class-Median Default Is Unbiased), plus known-answer / unit tests for the
boundary cases (theta=0, vertical-only), the type-specific path, the global
fallback, the worst-case-avoidance contract, and the
``class_default_widens_ci`` policy switch.
"""

import math
import statistics

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tdz.config.models import LeverArm
from tdz.config.schema import ClassMedian, LeverArmsConfig
from tdz.geo import (
    LeverArmCorrection,
    compute_lever_arm_correction,
    compute_lever_arm_range_widening,
    horizontal_ground_correction,
    resolve_lever_arm,
    resolve_lever_arm_correction,
)
from tdz.geo.errors import LeverArmResolutionError
from tdz.models import FailureReason

# Geometry equality is exact arithmetic; allow only floating-point slack.
_TOL_M = 1e-9


def _arm(
    icao_type: str,
    *,
    vertical: float,
    longitudinal: float,
    pitch: float,
    aircraft_class: str = "narrowbody",
    is_class_default: bool = False,
) -> LeverArm:
    """Build a LeverArm for tests."""
    return LeverArm(
        icao_type=icao_type,
        vertical_offset_m=vertical,
        longitudinal_offset_m=longitudinal,
        nominal_touchdown_pitch_deg=pitch,
        aircraft_class=aircraft_class,
        is_class_default=is_class_default,
    )


def _config(
    arms: dict[str, LeverArm],
    class_medians: dict[str, ClassMedian],
    *,
    widens: bool = True,
) -> LeverArmsConfig:
    """Build a LeverArmsConfig for tests."""
    return LeverArmsConfig(
        arms=arms,
        default_strategy="class_median",
        class_medians=class_medians,
        class_default_widens_ci=widens,
    )


# ---------------------------------------------------------------------------
# Property 2: Lever-Arm Correction Geometric Consistency
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    vertical=st.floats(min_value=0.0, max_value=10.0),
    longitudinal=st.floats(min_value=-30.0, max_value=30.0),
    pitch=st.floats(min_value=0.0, max_value=15.0),
)
def test_lever_arm_correction_geometric_consistency(vertical, longitudinal, pitch):
    """Feature: touchdown-point-detection, Property 2: Lever-Arm Correction Geometric Consistency

    For any vertical offset V, longitudinal offset X, and nominal pitch theta,
    the along-runway shift equals X*cos(theta) + V*sin(theta) (the full
    horizontal term, including the height-induced component) and the
    altitude-target shift equals exactly V. Pitch is recorded as assumed.
    """
    arm = _arm("TST", vertical=vertical, longitudinal=longitudinal, pitch=pitch)
    correction = compute_lever_arm_correction(arm)

    theta = math.radians(pitch)
    expected_along = longitudinal * math.cos(theta) + vertical * math.sin(theta)

    assert correction.along_runway_shift_m == pytest.approx(expected_along, abs=_TOL_M)
    assert correction.altitude_target_shift_m == pytest.approx(vertical, abs=_TOL_M)
    assert correction.assumed_pitch_deg == pytest.approx(pitch, abs=_TOL_M)
    # Pitch is assumed (per-type nominal), never measured (Req 7.3).
    assert correction.pitch_assumed is True
    # Type-specific arm -> normal confidence, no widening.
    assert correction.is_class_default is False
    assert correction.low_confidence is False
    assert correction.reason_code is None
    assert correction.ci_widening_m == 0.0


@pytest.mark.property
@given(
    vertical=st.floats(min_value=0.0, max_value=10.0),
    longitudinal=st.floats(min_value=-30.0, max_value=30.0),
)
def test_lever_arm_correction_pitch_zero_is_longitudinal(vertical, longitudinal):
    """Boundary theta=0: along-runway shift reduces to exactly X (Property 2)."""
    arm = _arm("TST", vertical=vertical, longitudinal=longitudinal, pitch=0.0)
    correction = compute_lever_arm_correction(arm)
    assert correction.along_runway_shift_m == pytest.approx(longitudinal, abs=_TOL_M)
    assert correction.altitude_target_shift_m == pytest.approx(vertical, abs=_TOL_M)


# ---------------------------------------------------------------------------
# Property 23: Class-Median Default Is Unbiased
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    members=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=10.0),     # vertical
            st.floats(min_value=-30.0, max_value=30.0),   # longitudinal
            st.floats(min_value=0.0, max_value=10.0),     # pitch
        ),
        min_size=3,
        max_size=7,
        unique_by=lambda t: t[1],  # distinct longitudinal offsets
    ),
)
def test_class_median_default_is_unbiased(members):
    """Feature: touchdown-point-detection, Property 23: Class-Median Default Is Unbiased

    For an aircraft type absent from the table, the applied default equals the
    median of its aircraft class, is flagged class-default + low-confidence with
    MISSING_LEVER_ARM, indicates a positive distance widening, and is never the
    largest-offset (worst-case) member.
    """
    aircraft_class = "narrowbody"
    arms = {
        f"M{i}": _arm(
            f"M{i}", vertical=v, longitudinal=x, pitch=p, aircraft_class=aircraft_class
        )
        for i, (v, x, p) in enumerate(members)
    }

    median_v = statistics.median(v for v, _, _ in members)
    median_x = statistics.median(x for _, x, _ in members)
    median_p = statistics.median(p for _, _, p in members)
    class_medians = {
        aircraft_class: ClassMedian(
            vertical_offset_m=median_v,
            longitudinal_offset_m=median_x,
            nominal_touchdown_pitch_deg=median_p,
        )
    }
    config = _config(arms, class_medians, widens=True)

    # Need a non-degenerate range for the widening to be strictly positive.
    member_corrections = [
        horizontal_ground_correction(x, v, p) for v, x, p in members
    ]
    assume(max(member_corrections) - min(member_corrections) > 1e-6)

    # "ABSENT" is not a key in arms.
    result = resolve_lever_arm_correction("ABSENT", config, aircraft_class)

    # Applied default equals the class median (the statistical median values).
    assert result.is_class_default is True
    assert result.lever_arm.vertical_offset_m == pytest.approx(median_v, abs=_TOL_M)
    assert result.lever_arm.longitudinal_offset_m == pytest.approx(median_x, abs=_TOL_M)
    assert result.lever_arm.nominal_touchdown_pitch_deg == pytest.approx(
        median_p, abs=_TOL_M
    )
    assert result.aircraft_class == aircraft_class

    # Low-confidence with the missing-lever-arm reason (Req 7.4a).
    assert result.low_confidence is True
    assert result.reason_code is FailureReason.MISSING_LEVER_ARM

    # Distance widening spans the class lever-arm range and is positive (Req 7.4b).
    assert result.ci_widening_m > 0.0
    assert result.ci_widening_m == pytest.approx(
        max(member_corrections) - min(member_corrections), abs=_TOL_M
    )

    # NOT a worst-case default: the median offset lies within the member range
    # and never exceeds the largest-offset member (Req 7.5).
    max_long = max(x for _, x, _ in members)
    min_long = min(x for _, x, _ in members)
    assert min_long <= result.lever_arm.longitudinal_offset_m <= max_long


# ---------------------------------------------------------------------------
# Known-answer / unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_type_specific_entry_used_as_is():
    """A type present in the table is used as-is: no default, no widening."""
    arm = _arm("B738", vertical=3.0, longitudinal=12.0, pitch=5.0)
    config = _config(
        {"B738": arm},
        {"narrowbody": ClassMedian(2.0, 8.0, 4.0)},
    )
    result = resolve_lever_arm_correction("B738", config, "narrowbody")
    assert result.is_class_default is False
    assert result.low_confidence is False
    assert result.reason_code is None
    assert result.ci_widening_m == 0.0
    assert result.lever_arm is arm


@pytest.mark.unit
def test_resolve_lever_arm_returns_type_specific_without_class():
    """Type-specific resolution does not require an aircraft_class argument."""
    arm = _arm("A320", vertical=3.5, longitudinal=14.0, pitch=6.0)
    config = _config({"A320": arm}, {})
    resolved = resolve_lever_arm("A320", config)
    assert resolved is arm


@pytest.mark.unit
def test_pitch_zero_known_answer():
    """theta=0 -> along-runway shift equals X exactly."""
    arm = _arm("TST", vertical=4.0, longitudinal=10.0, pitch=0.0)
    correction = compute_lever_arm_correction(arm)
    assert correction.along_runway_shift_m == pytest.approx(10.0, abs=_TOL_M)
    assert correction.altitude_target_shift_m == pytest.approx(4.0, abs=_TOL_M)


@pytest.mark.unit
def test_vertical_only_known_answer():
    """X=0 -> along shift = V*sin(theta); altitude shift = V."""
    vertical, pitch = 5.0, 6.0
    arm = _arm("TST", vertical=vertical, longitudinal=0.0, pitch=pitch)
    correction = compute_lever_arm_correction(arm)
    expected_along = vertical * math.sin(math.radians(pitch))
    assert correction.along_runway_shift_m == pytest.approx(expected_along, abs=_TOL_M)
    assert correction.altitude_target_shift_m == pytest.approx(vertical, abs=_TOL_M)
    # The along shift is purely the height-induced term, not zero.
    assert correction.along_runway_shift_m > 0.0


@pytest.mark.unit
def test_class_median_not_worst_case_known_answer():
    """Explicit median-vs-max check: the default is the median, not the largest.

    Three narrowbody members with longitudinal offsets {8, 12, 20}: the class
    median is 12, never the worst-case 20 (Req 7.5 / Property 23).
    """
    arms = {
        "S": _arm("S", vertical=2.0, longitudinal=8.0, pitch=4.0),
        "M": _arm("M", vertical=3.0, longitudinal=12.0, pitch=5.0),
        "L": _arm("L", vertical=4.0, longitudinal=20.0, pitch=6.0),
    }
    class_medians = {"narrowbody": ClassMedian(3.0, 12.0, 5.0)}
    config = _config(arms, class_medians)
    result = resolve_lever_arm_correction("ABSENT", config, "narrowbody")
    assert result.lever_arm.longitudinal_offset_m == pytest.approx(12.0, abs=_TOL_M)
    assert result.lever_arm.longitudinal_offset_m != 20.0  # not worst-case
    assert result.is_class_default is True
    assert result.ci_widening_m > 0.0


@pytest.mark.unit
def test_global_median_fallback_when_class_unknown():
    """Unknown class -> global median across all table entries, class 'unknown'."""
    arms = {
        "A": _arm("A", vertical=2.0, longitudinal=8.0, pitch=4.0, aircraft_class="narrowbody"),
        "B": _arm("B", vertical=4.0, longitudinal=12.0, pitch=6.0, aircraft_class="widebody"),
        "C": _arm("C", vertical=6.0, longitudinal=16.0, pitch=8.0, aircraft_class="widebody"),
    }
    config = _config(arms, {})  # no class medians at all
    # No aircraft_class supplied -> global median path.
    result = resolve_lever_arm_correction("ABSENT", config, None)
    assert result.is_class_default is True
    assert result.aircraft_class == "unknown"
    assert result.reason_code is FailureReason.MISSING_LEVER_ARM
    # Global medians across {2,4,6}, {8,12,16}, {4,6,8} -> 4, 12, 6.
    assert result.lever_arm.vertical_offset_m == pytest.approx(4.0, abs=_TOL_M)
    assert result.lever_arm.longitudinal_offset_m == pytest.approx(12.0, abs=_TOL_M)
    assert result.lever_arm.nominal_touchdown_pitch_deg == pytest.approx(6.0, abs=_TOL_M)


@pytest.mark.unit
def test_unknown_class_with_class_medians_falls_back_to_global():
    """A class with no configured median falls back to the global median."""
    arms = {
        "A": _arm("A", vertical=2.0, longitudinal=8.0, pitch=4.0, aircraft_class="narrowbody"),
        "B": _arm("B", vertical=4.0, longitudinal=12.0, pitch=6.0, aircraft_class="narrowbody"),
    }
    # class_medians has only "widebody"; requesting "regional" misses it.
    config = _config(arms, {"widebody": ClassMedian(10.0, 20.0, 9.0)})
    result = resolve_lever_arm_correction("ABSENT", config, "regional")
    assert result.is_class_default is True
    assert result.aircraft_class == "unknown"
    # Global medians across the two narrowbody entries: {2,4}->3, {8,12}->10, {4,6}->5.
    assert result.lever_arm.vertical_offset_m == pytest.approx(3.0, abs=_TOL_M)
    assert result.lever_arm.longitudinal_offset_m == pytest.approx(10.0, abs=_TOL_M)
    assert result.lever_arm.nominal_touchdown_pitch_deg == pytest.approx(5.0, abs=_TOL_M)


@pytest.mark.unit
def test_widening_suppressed_when_policy_off_but_low_confidence_kept():
    """class_default_widens_ci=False suppresses widening, keeps low-confidence."""
    arms = {
        "S": _arm("S", vertical=2.0, longitudinal=8.0, pitch=4.0),
        "L": _arm("L", vertical=4.0, longitudinal=20.0, pitch=6.0),
    }
    class_medians = {"narrowbody": ClassMedian(3.0, 14.0, 5.0)}
    config = _config(arms, class_medians, widens=False)
    result = resolve_lever_arm_correction("ABSENT", config, "narrowbody")
    assert result.is_class_default is True
    assert result.low_confidence is True
    assert result.reason_code is FailureReason.MISSING_LEVER_ARM
    assert result.ci_widening_m == 0.0


@pytest.mark.unit
def test_resolution_error_when_nothing_available():
    """No type entry, no class median, empty table -> LeverArmResolutionError."""
    config = _config({}, {})
    with pytest.raises(LeverArmResolutionError) as exc_info:
        resolve_lever_arm_correction("ABSENT", config, "narrowbody")
    assert exc_info.value.reason_code is FailureReason.MISSING_LEVER_ARM


@pytest.mark.unit
def test_range_widening_single_arm_is_zero():
    """Spread of a single lever arm is undefined -> 0.0."""
    arm = _arm("ONE", vertical=3.0, longitudinal=12.0, pitch=5.0)
    assert compute_lever_arm_range_widening([arm]) == 0.0
    assert compute_lever_arm_range_widening([]) == 0.0


@pytest.mark.unit
def test_correction_is_frozen():
    """LeverArmCorrection is an immutable value object."""
    arm = _arm("TST", vertical=3.0, longitudinal=12.0, pitch=5.0)
    correction = compute_lever_arm_correction(arm)
    assert isinstance(correction, LeverArmCorrection)
    with pytest.raises(Exception):
        correction.along_runway_shift_m = 99.0  # type: ignore[misc]


@pytest.mark.unit
def test_inputs_not_mutated():
    """Computing a correction does not mutate the input lever arm."""
    arm = _arm("TST", vertical=3.0, longitudinal=12.0, pitch=5.0)
    compute_lever_arm_correction(arm)
    assert arm.vertical_offset_m == 3.0
    assert arm.longitudinal_offset_m == 12.0
    assert arm.nominal_touchdown_pitch_deg == 5.0
    assert arm.is_class_default is False
