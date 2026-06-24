"""Tests for trajectory classification & the coarse bracket (Task 10, Module 1b).

Covers Property 21 (Go-Around Produces No Touchdown, plus the bounce
first-contact anchoring half of the property) and known-answer unit tests for
:func:`compute_coarse_bracket` / :func:`classify_trajectory`:

* clean completed landing -> ``"ok"`` window straddling the touchdown;
* on-ground flag is an UPPER bound only (``t_hi <= on_ground_transition_time``);
* a bracket still forms with NO on-ground flag (flag-independent indicators);
* touch-and-go report-vs-suppress policy;
* multiple-landing detection (``n_landings >= 2``);
* datum-unresolved degradation (relative-height classification still works);
* the classification confusion matrix (Req 21.6);
* integration: a real bracket window feeds straight into
  :func:`tdz.io.run_qa` (closing the Task 9 chicken-and-egg).

Synthetic flights are built at a 4-5 s cadence with explicit per-sample heights
above the runway threshold; the runway uses ``elevation_datum="HAE"`` (geometric
altitude and threshold elevation directly comparable) except where a test
deliberately exercises MSL datum resolution / degradation.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.bracket import (
    CLIMB_OUT_HEIGHT_M,
    CONTACT_HEIGHT_M,
    DEFAULT_BRACKET_HALF_WIDTH_S,
    INDICATOR_ALTITUDE_DESCENT,
    INDICATOR_DECEL_ONSET,
    INDICATOR_ON_GROUND_FLAG,
    TRAJECTORY_COMPLETED_LANDING,
    TRAJECTORY_GO_AROUND,
    TRAJECTORY_TOUCH_AND_GO,
    TRAJECTORY_TYPES,
    classification_confusion_matrix,
    classify_trajectory,
    compute_coarse_bracket,
)
from tdz.config.schema import QualityGatesConfig
from tdz.io import TouchdownWindow, run_qa
from tdz.models import FailureReason, FlightRecord, RunwayReference


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _runway(
    *,
    elevation_m: float = 100.0,
    datum: str = "HAE",
    undulation: float = 0.0,
    heading: float = 250.0,
    lat: float = 33.94,
    lon: float = -118.40,
) -> RunwayReference:
    return RunwayReference(
        threshold_lat=lat,
        threshold_lon=lon,
        heading_deg=heading,
        elevation_m=elevation_m,
        elevation_datum=datum,
        geoid_undulation_m=undulation,
        length_m=3000.0,
        width_m=45.0,
        displaced=False,
    )


def _threshold_hae(runway: RunwayReference) -> float:
    """The HAE threshold elevation a test-built geometric altitude is relative to."""
    if runway.elevation_datum == "HAE":
        return runway.elevation_m
    return runway.elevation_m + runway.geoid_undulation_m


def _gates(**overrides) -> QualityGatesConfig:
    base = dict(
        min_samples_near_td=3,
        max_gap_spanning_td_s=15.0,
        min_samples_in_window=3,
        window_half_width_s=30.0,
        max_excluded_fraction=0.5,
        max_longitudinal_accel_g=1.0,
        max_lateral_accel_g=0.5,
        max_turn_rate_deg_s=6.0,
        duplicate_timestamp_tolerance_s=0.1,
    )
    base.update(overrides)
    return QualityGatesConfig(**base)


def _make_flight(
    *,
    times,
    heights_above_runway,
    runway: RunwayReference,
    on_ground_flags,
    groundspeeds,
    on_ground_transition_time,
    lat0: float = 33.9,
    lon0: float = -118.4,
    heading: float = 250.0,
    velocity_times=None,
    flight_id: str = "T",
    aircraft_type: str = "B738",
    source: str = "aireon",
) -> FlightRecord:
    """Assemble a :class:`FlightRecord` from explicit per-sample heights.

    ``heights_above_runway`` are metres above the (HAE-resolved) threshold; the
    stored ``geometric_altitudes`` are ``threshold_hae + height`` so the
    classifier sees exactly the intended height profile.
    """
    times = np.asarray(times, dtype=float)
    heights = np.asarray(heights_above_runway, dtype=float)
    n = times.size
    geo_alt = _threshold_hae(runway) + heights
    lat = lat0 + 1e-4 * np.arange(n)
    lon = lon0 + 1e-4 * np.arange(n)
    tracks = np.full(n, heading, dtype=float)
    baro = np.full(n, np.nan, dtype=float)
    vr = np.full(n, np.nan, dtype=float)
    gs = np.asarray(groundspeeds, dtype=float)
    vt = times if velocity_times is None else np.asarray(velocity_times, dtype=float)
    flags = np.asarray(on_ground_flags, dtype=bool)
    return FlightRecord(
        flight_id=flight_id,
        aircraft_type=aircraft_type,
        ads_b_source=source,
        position_times=times,
        velocity_times=vt,
        latitudes=lat,
        longitudes=lon,
        geometric_altitudes=geo_alt,
        barometric_altitudes=baro,
        groundspeeds=gs,
        tracks=tracks,
        baro_vertical_rates=vr,
        on_ground_flags=flags,
        on_ground_transition_time=on_ground_transition_time,
        runway=runway,
    )


def _clean_landing_flight(
    *,
    runway: RunwayReference | None = None,
    td_step: int = 5,
    roll_steps: int = 6,
    dt: float = 4.0,
    on_ground: bool = True,
    flag_offset_steps: int = 2,
) -> tuple[FlightRecord, float, float | None]:
    """A clean completed landing: steep descent to contact, then ground roll.

    Returns ``(flight, t_touchdown, on_ground_transition_time)`` where
    ``t_touchdown`` is the first-contact sample time (height 0).
    """
    runway = runway if runway is not None else _runway()
    n = td_step + roll_steps + 1
    times = np.arange(n, dtype=float) * dt
    heights = np.empty(n, dtype=float)
    # Steep approach 300 m -> 0 over td_step samples (big steps; no sub-5 m tail).
    heights[: td_step + 1] = np.linspace(300.0, 0.0, td_step + 1)
    heights[td_step + 1 :] = 0.0  # sustained ground roll at the surface
    t_td = float(times[td_step])
    flags = np.zeros(n, dtype=bool)
    transition = None
    if on_ground:
        flags[td_step + flag_offset_steps :] = True
        transition = float(times[td_step + flag_offset_steps])
    # Gentle deceleration through and after touchdown.
    gs = np.clip(150.0 - 3.0 * np.arange(n, dtype=float), 40.0, None)
    flight = _make_flight(
        times=times,
        heights_above_runway=heights,
        runway=runway,
        on_ground_flags=flags,
        groundspeeds=gs,
        on_ground_transition_time=transition,
    )
    return flight, t_td, transition


# ---------------------------------------------------------------------------
# Property 21: Go-Around Produces No Touchdown (+ bounce first-contact anchor)
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    descent_steps=st.integers(min_value=3, max_value=6),
    climb_steps=st.integers(min_value=3, max_value=8),
    h0=st.floats(min_value=200.0, max_value=360.0),
    min_h=st.floats(min_value=10.0, max_value=40.0),
    peak_h=st.floats(min_value=120.0, max_value=300.0),
    gs0=st.floats(min_value=120.0, max_value=160.0),
    dt=st.sampled_from([4.0, 5.0]),
    lat0=st.floats(min_value=-60.0, max_value=60.0),
    lon0=st.floats(min_value=-179.0, max_value=179.0),
    heading=st.floats(min_value=0.0, max_value=359.0),
)
def test_p21_go_around_produces_no_touchdown(
    descent_steps, climb_steps, h0, min_h, peak_h, gs0, dt, lat0, lon0, heading
):
    """Feature: touchdown-point-detection, Property 21: Go-Around Produces No Touchdown

    An approach that descends toward the runway but never reaches contact
    (minimum height stays above ``CONTACT_HEIGHT_M``) and then climbs back out
    well above ``CLIMB_OUT_HEIGHT_M`` is classified as a go-around: no contact
    segments, reason ``GO_AROUND``, ``is_touchdown=False``. The coarse bracket
    short-circuits to ``"no-touchdown"`` with no window (Req 21.2; design
    Property 21).
    """
    assert min_h > CONTACT_HEIGHT_M  # never makes contact
    assert peak_h > CLIMB_OUT_HEIGHT_M  # genuinely climbs out

    runway = _runway(heading=heading)
    descent = np.linspace(h0, min_h, descent_steps + 1)
    climb = np.linspace(min_h, peak_h, climb_steps + 1)[1:]
    heights = np.concatenate([descent, climb])
    n = heights.size
    times = np.arange(n, dtype=float) * dt
    flags = np.zeros(n, dtype=bool)  # never on the ground
    gs = np.full(n, gs0, dtype=float)

    flight = _make_flight(
        times=times,
        heights_above_runway=heights,
        runway=runway,
        on_ground_flags=flags,
        groundspeeds=gs,
        on_ground_transition_time=None,
        lat0=lat0,
        lon0=lon0,
        heading=heading,
    )

    cls = classify_trajectory(flight)
    assert cls.trajectory_type == TRAJECTORY_GO_AROUND
    assert cls.reason_code is FailureReason.GO_AROUND
    assert cls.is_touchdown is False
    assert cls.contacts == ()

    bracket = compute_coarse_bracket(flight)
    assert bracket.status == "no-touchdown"
    assert bracket.window is None
    assert bracket.reason_code is FailureReason.GO_AROUND


@pytest.mark.property
@given(
    descent_steps=st.integers(min_value=2, max_value=5),
    gap_steps=st.integers(min_value=2, max_value=6),
    roll_steps=st.integers(min_value=3, max_value=5),
    h0=st.floats(min_value=200.0, max_value=360.0),
    bounce_peak=st.floats(min_value=15.0, max_value=50.0),
    gs0=st.floats(min_value=120.0, max_value=160.0),
    dt=st.sampled_from([4.0, 5.0]),
    lat0=st.floats(min_value=-60.0, max_value=60.0),
    lon0=st.floats(min_value=-179.0, max_value=179.0),
    heading=st.floats(min_value=0.0, max_value=359.0),
)
def test_p21_bounce_brackets_first_contact_not_midpoint(
    descent_steps, gap_steps, roll_steps, h0, bounce_peak, gs0, dt, lat0, lon0, heading
):
    """Feature: touchdown-point-detection, Property 21: Go-Around Produces No Touchdown

    A bounce -- a first contact at ``t1``, a brief re-rise below
    ``CLIMB_OUT_HEIGHT_M``, a second contact at ``t2 > t1``, then sustained
    ground roll -- is a completed landing whose bracket is anchored to the
    FIRST contact. The reported first-contact time is at/near ``t1`` (within one
    sample, reflecting the interpolated crossing) and is never a value strictly
    between ``t1`` and ``t2`` (never the average of the two contacts) (Req 21.4).
    """
    assert CONTACT_HEIGHT_M < bounce_peak < CLIMB_OUT_HEIGHT_M

    runway = _runway(heading=heading)
    i1 = descent_steps
    i2 = i1 + gap_steps
    n = i2 + roll_steps + 1
    heights = np.zeros(n, dtype=float)
    # Steep approach to the first contact at i1 (height 0).
    heights[: i1 + 1] = np.linspace(h0, 0.0, i1 + 1)
    # Triangular bounce i1 -> i2: up to bounce_peak at the midpoint, back to 0.
    for j in range(1, gap_steps):
        frac = j / gap_steps
        heights[i1 + j] = bounce_peak * (1.0 - abs(2.0 * frac - 1.0))
    heights[i2:] = 0.0  # sustained ground roll after the second contact

    times = np.arange(n, dtype=float) * dt
    t1 = float(times[i1])
    t2 = float(times[i2])

    flags = np.zeros(n, dtype=bool)
    flags[i2:] = True  # flag confirms once the sustained ground roll starts
    transition = t2
    gs = np.clip(gs0 - 2.0 * np.arange(n, dtype=float), 30.0, None)

    flight = _make_flight(
        times=times,
        heights_above_runway=heights,
        runway=runway,
        on_ground_flags=flags,
        groundspeeds=gs,
        on_ground_transition_time=transition,
        lat0=lat0,
        lon0=lon0,
        heading=heading,
    )

    cls = classify_trajectory(flight)
    assert cls.is_touchdown is True
    assert cls.trajectory_type == TRAJECTORY_COMPLETED_LANDING
    assert len(cls.contacts) >= 2  # the bounce is two distinct contact segments

    bracket = compute_coarse_bracket(flight)
    assert bracket.status == "ok"
    assert bracket.window is not None
    center = bracket.window.center
    midpoint = 0.5 * (t1 + t2)

    # Anchored at/near the FIRST contact: the interpolated crossing lands within
    # one sample at or before t1, never near the midpoint of the two contacts.
    assert t1 - dt - 1e-6 <= center <= t1 + 1e-6
    assert center < midpoint
    assert center == pytest.approx(bracket.first_contact_time)


# ---------------------------------------------------------------------------
# Known-answer unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clean_completed_landing_window_contains_touchdown():
    """A clean landing yields status 'ok' with t_lo < t_hi straddling touchdown.

    Uses an MSL runway with a valid geoid undulation to exercise datum
    resolution (geometric altitude compared against the geoid-corrected HAE
    threshold elevation).
    """
    runway = _runway(elevation_m=38.0, datum="MSL", undulation=-35.0)
    flight, t_td, transition = _clean_landing_flight(runway=runway)

    cls = classify_trajectory(flight)
    assert cls.datum_resolved is True
    assert cls.trajectory_type == TRAJECTORY_COMPLETED_LANDING
    assert cls.is_touchdown is True

    bracket = compute_coarse_bracket(flight)
    assert bracket.status == "ok"
    assert bracket.window is not None
    assert bracket.window.t_lo < bracket.window.t_hi
    # The known touchdown time lies inside the bracket.
    assert bracket.window.t_lo <= t_td <= bracket.window.t_hi
    assert bracket.window.contains(bracket.window.center)


@pytest.mark.unit
def test_on_ground_flag_is_upper_bound_only():
    """The on-ground flag transition caps t_hi (flag lags the real touchdown)."""
    flight, _t_td, transition = _clean_landing_flight()
    assert transition is not None

    bracket = compute_coarse_bracket(flight)
    assert bracket.status == "ok"
    assert bracket.window is not None
    assert bracket.on_ground_upper_bound == transition
    # t_hi never exceeds the on-ground transition time (Req 18.2).
    assert bracket.window.t_hi <= transition + 1e-9
    assert INDICATOR_ON_GROUND_FLAG in bracket.indicators_fired


@pytest.mark.unit
def test_bracket_forms_without_on_ground_flag():
    """With no on-ground flag a bracket still forms from the flag-independent indicators."""
    flight, _t_td, _transition = _clean_landing_flight(on_ground=False)
    assert flight.on_ground_transition_time is None
    assert not flight.on_ground_flags.any()

    bracket = compute_coarse_bracket(flight)
    assert bracket.status == "ok"
    assert bracket.window is not None
    assert bracket.on_ground_upper_bound is None
    # A flag-independent indicator carried the bracket.
    assert (
        INDICATOR_ALTITUDE_DESCENT in bracket.indicators_fired
        or INDICATOR_DECEL_ONSET in bracket.indicators_fired
    )
    assert INDICATOR_ON_GROUND_FLAG not in bracket.indicators_fired


def _touch_and_go_flight() -> tuple[FlightRecord, float]:
    """Brief contact then a climb-out above CLIMB_OUT_HEIGHT_M (no ground roll)."""
    runway = _runway()
    dt = 4.0
    # Descend 300 -> 0 (contact at idx 3), then climb to 220 m (>> climb-out).
    heights = np.array([300.0, 200.0, 100.0, 0.0, 90.0, 180.0, 220.0], dtype=float)
    n = heights.size
    times = np.arange(n, dtype=float) * dt
    tc = float(times[3])
    flags = np.zeros(n, dtype=bool)
    flags[3] = True  # a single on-ground sample at the brief contact
    gs = np.array([150.0, 145.0, 140.0, 138.0, 142.0, 150.0, 160.0], dtype=float)
    flight = _make_flight(
        times=times,
        heights_above_runway=heights,
        runway=runway,
        on_ground_flags=flags,
        groundspeeds=gs,
        on_ground_transition_time=tc,
    )
    return flight, tc


@pytest.mark.unit
def test_touch_and_go_report_policy_produces_bracket():
    """Policy 'report' tags TOUCH_AND_GO yet still brackets the brief contact."""
    flight, _tc = _touch_and_go_flight()

    cls = classify_trajectory(flight, touch_and_go_policy="report")
    assert cls.trajectory_type == TRAJECTORY_TOUCH_AND_GO
    assert cls.reason_code is FailureReason.TOUCH_AND_GO
    assert cls.is_touchdown is True

    bracket = compute_coarse_bracket(flight, touch_and_go_policy="report")
    assert bracket.status == "ok"
    assert bracket.window is not None
    assert bracket.trajectory_type == TRAJECTORY_TOUCH_AND_GO


@pytest.mark.unit
def test_touch_and_go_suppress_policy_no_touchdown():
    """Policy 'suppress' emits a no-touchdown TOUCH_AND_GO result (no window)."""
    flight, _tc = _touch_and_go_flight()

    cls = classify_trajectory(flight, touch_and_go_policy="suppress")
    assert cls.trajectory_type == TRAJECTORY_TOUCH_AND_GO
    assert cls.is_touchdown is False

    bracket = compute_coarse_bracket(flight, touch_and_go_policy="suppress")
    assert bracket.status == "no-touchdown"
    assert bracket.window is None
    assert bracket.reason_code is FailureReason.TOUCH_AND_GO


@pytest.mark.unit
def test_multiple_landings_detected_first_contact_targets_first():
    """Two landings separated by a full climb-out -> multiple_landings, first anchor."""
    runway = _runway()
    dt = 4.0
    # Land (contact idx 2..4), climb out above 60 m (idx 5..7), land again (idx 9..11).
    heights = np.array(
        [200.0, 100.0, 0.0, 0.0, 0.0, 120.0, 220.0, 120.0, 80.0, 0.0, 0.0, 0.0],
        dtype=float,
    )
    n = heights.size
    times = np.arange(n, dtype=float) * dt
    t1 = float(times[2])  # first landing's first contact
    flags = np.zeros(n, dtype=bool)
    flags[2:5] = True
    flags[9:] = True
    gs = np.full(n, 140.0, dtype=float)
    flight = _make_flight(
        times=times,
        heights_above_runway=heights,
        runway=runway,
        on_ground_flags=flags,
        groundspeeds=gs,
        on_ground_transition_time=float(times[2]),
    )

    cls = classify_trajectory(flight)
    assert cls.multiple_landings is True
    assert cls.n_landings >= 2
    # The reported first-contact time targets the FIRST landing.
    assert cls.first_contact_time is not None
    assert t1 - dt - 1e-6 <= cls.first_contact_time <= t1 + 1e-6


@pytest.mark.unit
def test_datum_unresolved_degrades_to_relative_classification():
    """An MSL runway with NaN undulation classifies via relative height + flag.

    The datum cannot be resolved (no finite geoid undulation), so the classifier
    degrades to a relative-altitude test; a result is still produced with
    ``datum_resolved=False`` and a bracket still forms from the surviving
    anchors (on-ground flag / deceleration onset).
    """
    runway = _runway(elevation_m=38.0, datum="MSL", undulation=float("nan"))
    dt = 4.0
    # Absolute HAE altitudes (relative profile = alt - min); min at the ground roll.
    heights = np.array([300.0, 200.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    n = heights.size
    times = np.arange(n, dtype=float) * dt
    flags = np.zeros(n, dtype=bool)
    flags[4:] = True
    gs = np.clip(150.0 - 5.0 * np.arange(n, dtype=float), 30.0, None)
    flight = _make_flight(
        times=times,
        heights_above_runway=heights,
        runway=runway,
        on_ground_flags=flags,
        groundspeeds=gs,
        on_ground_transition_time=float(times[4]),
    )

    cls = classify_trajectory(flight)
    assert cls.datum_resolved is False
    assert cls.is_touchdown is True  # still classifiable without an absolute datum
    assert cls.trajectory_type == TRAJECTORY_COMPLETED_LANDING

    bracket = compute_coarse_bracket(flight)
    assert bracket.datum_resolved is False
    assert bracket.status == "ok"
    assert bracket.window is not None
    # The absolute altitude-descent indicator is unavailable without a datum.
    assert INDICATOR_ALTITUDE_DESCENT not in bracket.indicators_fired


# ---------------------------------------------------------------------------
# Confusion matrix (Req 21.6)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confusion_matrix_counts_and_diagonal():
    """Nested counts match; the diagonal is the correctly classified count."""
    truth = [
        TRAJECTORY_COMPLETED_LANDING,
        TRAJECTORY_COMPLETED_LANDING,
        TRAJECTORY_GO_AROUND,
        TRAJECTORY_TOUCH_AND_GO,
    ]
    predicted = [
        TRAJECTORY_COMPLETED_LANDING,  # correct
        TRAJECTORY_GO_AROUND,          # completed misread as go-around
        TRAJECTORY_GO_AROUND,          # correct
        TRAJECTORY_TOUCH_AND_GO,       # correct
    ]
    matrix = classification_confusion_matrix(predicted, truth)

    # Every label appears as both a truth row and a predicted column.
    assert set(matrix) == set(TRAJECTORY_TYPES)
    for row in matrix.values():
        assert set(row) == set(TRAJECTORY_TYPES)

    # Diagonal = correct.
    assert matrix[TRAJECTORY_COMPLETED_LANDING][TRAJECTORY_COMPLETED_LANDING] == 1
    assert matrix[TRAJECTORY_GO_AROUND][TRAJECTORY_GO_AROUND] == 1
    assert matrix[TRAJECTORY_TOUCH_AND_GO][TRAJECTORY_TOUCH_AND_GO] == 1
    # The one off-diagonal: a completed landing predicted as a go-around.
    assert matrix[TRAJECTORY_COMPLETED_LANDING][TRAJECTORY_GO_AROUND] == 1
    # Total count is conserved.
    assert sum(c for row in matrix.values() for c in row.values()) == len(truth)


@pytest.mark.unit
def test_confusion_matrix_unequal_length_raises():
    """Predicted and truth must be equal length."""
    with pytest.raises(ValueError):
        classification_confusion_matrix(
            [TRAJECTORY_COMPLETED_LANDING], [TRAJECTORY_COMPLETED_LANDING, TRAJECTORY_GO_AROUND]
        )


@pytest.mark.unit
def test_confusion_matrix_unknown_label_raises():
    """Labels outside the label universe are rejected."""
    with pytest.raises(ValueError):
        classification_confusion_matrix(["banana"], [TRAJECTORY_GO_AROUND])
    with pytest.raises(ValueError):
        classification_confusion_matrix([TRAJECTORY_GO_AROUND], ["banana"])


# ---------------------------------------------------------------------------
# Integration: the real bracket window feeds straight into run_qa (Task 9 link)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bracket_window_feeds_run_qa():
    """A coarse-bracket window is accepted by run_qa's sufficiency gate.

    Closes the chicken-and-egg the QA module documented: the bracket is emitted
    as the very :class:`~tdz.io.qa.TouchdownWindow` the sufficiency gate consumes.
    """
    runway = _runway()
    dt = 4.0
    n = 16
    times = np.arange(n, dtype=float) * dt
    # Descend to contact around idx 8, then ground roll to the end.
    heights = np.concatenate(
        [np.linspace(320.0, 0.0, 9), np.zeros(n - 9, dtype=float)]
    )
    flags = np.zeros(n, dtype=bool)
    flags[10:] = True
    gs = np.clip(150.0 - 3.0 * np.arange(n, dtype=float), 40.0, None)
    flight = _make_flight(
        times=times,
        heights_above_runway=heights,
        runway=runway,
        on_ground_flags=flags,
        groundspeeds=gs,
        on_ground_transition_time=float(times[10]),
    )

    bracket = compute_coarse_bracket(flight, half_width_s=30.0)
    assert bracket.status == "ok"
    assert isinstance(bracket.window, TouchdownWindow)

    qa = run_qa(flight, _gates(min_samples_in_window=3), touchdown_window=bracket.window)
    # The status reflects the (sufficient) data, not a window error.
    assert qa.status == "ok"
    assert qa.reason_code is None
    assert qa.diagnostics.sufficiency is not None
    assert qa.diagnostics.sufficiency.window is bracket.window


@pytest.mark.unit
def test_default_bracket_half_width_constant():
    """The default coarse-bracket half-width matches the documented QA default."""
    assert DEFAULT_BRACKET_HALF_WIDTH_S == 30.0
