"""Tests for the QA-gate module (Task 9.3).

Covers Property 8 (Duplicate Timestamp Deduplication), Property 9 (Kinematic
Gate Exclusion) and Property 13 (Missing Vertical Rate Tolerance), plus
known-answer unit tests: dedup keeps last-received; a 2-g sample is excluded
with its timestamp logged; a 16 s gap spanning the window -> GAP_SPANS_TOUCHDOWN;
>50% excluded -> EXCESSIVE_EXCLUSIONS; missing groundspeed -> NO_GROUNDSPEED;
and the standard-gravity constant.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.config.schema import QualityGatesConfig
from tdz.io import (
    STANDARD_GRAVITY_MPS2,
    TouchdownWindow,
    apply_kinematic_gates,
    deduplicate_by_timestamp,
    evaluate_sufficiency,
    parse_fr24_records,
    run_qa,
)
from tdz.models import FailureReason, FR24Record, RunwayReference
from tdz.timebase import KNOTS_TO_MPS


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


def _runway() -> RunwayReference:
    return RunwayReference(
        threshold_lat=33.94, threshold_lon=-118.40, heading_deg=250.0,
        elevation_m=38.0, elevation_datum="MSL", geoid_undulation_m=-35.0,
        length_m=3000.0, width_m=45.0, displaced=False,
    )


# ---------------------------------------------------------------------------
# Property 8: Duplicate Timestamp Deduplication
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    unique_times=st.lists(
        st.integers(min_value=0, max_value=500),
        min_size=1,
        max_size=20,
        unique=True,
    ),
    dup_counts=st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=20),
)
def test_duplicate_dedup_counts_and_last_received(unique_times, dup_counts):
    """Feature: touchdown-point-detection, Property 8: Duplicate Timestamp Deduplication

    For N samples forming K unique-timestamp groups (each repeated 1..4 times),
    deduplication yields exactly N - total_duplicates samples (= K), one per
    unique timestamp, retaining the last-received (greatest input index) in each
    group (Req 9.3).
    """
    # Build groups: each unique timestamp repeated dup_counts[i] times.
    times: list[float] = []
    group_of: list[int] = []  # which unique-timestamp group each sample belongs to
    for gi, ut in enumerate(unique_times):
        reps = dup_counts[gi % len(dup_counts)]
        for _ in range(reps):
            times.append(float(ut))
            group_of.append(gi)

    # Shuffle deterministically by interleaving so duplicates are not adjacent.
    order = sorted(range(len(times)), key=lambda i: (i * 7) % len(times))
    times_arr = np.array([times[i] for i in order], dtype=float)
    group_arr = [group_of[i] for i in order]

    n = times_arr.size
    k = len(unique_times)
    total_duplicates = n - k

    result = deduplicate_by_timestamp(times_arr, tolerance_s=0.1)

    # Exactly N - total_duplicates == K kept, one per unique timestamp.
    assert result.n_kept == n - total_duplicates == k
    assert result.n_removed == total_duplicates
    kept_times = sorted(times_arr[result.kept_indices].tolist())
    assert kept_times == sorted(float(u) for u in unique_times)

    # Last-received retained: for each kept index, it is the max input index in
    # its group (the latest-arriving duplicate).
    for kept_idx in result.kept_indices.tolist():
        g = group_arr[kept_idx]
        group_indices = [i for i in range(n) if group_arr[i] == g]
        assert kept_idx == max(group_indices)


@pytest.mark.unit
def test_dedup_keeps_last_received_concrete():
    """Concrete example: among exact duplicates the last-received index wins."""
    # Indices 0,1,2 share t=5.0; index 3 is t=9.0.
    times = np.array([5.0, 5.0, 5.0, 9.0])
    result = deduplicate_by_timestamp(times, tolerance_s=0.1)
    assert result.n_kept == 2
    assert result.kept_indices.tolist() == [2, 3]  # last of the t=5 group, plus t=9
    assert result.removed_timestamps == (5.0, 5.0)


@pytest.mark.unit
def test_dedup_near_duplicate_within_tolerance():
    """Timestamps within tolerance collapse; outside tolerance are distinct."""
    times = np.array([10.00, 10.05, 10.20])  # 0.05 within tol; 0.15 apart -> distinct
    result = deduplicate_by_timestamp(times, tolerance_s=0.1)
    # 10.00 & 10.05 cluster (last-received idx 1); 10.20 separate.
    assert result.n_kept == 2
    assert result.kept_indices.tolist() == [1, 2]


# ---------------------------------------------------------------------------
# Property 9: Kinematic Gate Exclusion
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n=st.integers(min_value=6, max_value=20),
    base_gs=st.floats(min_value=120.0, max_value=160.0),
    inject_idx=st.integers(min_value=1, max_value=18),
)
def test_kinematic_gate_excludes_and_leaves_plausible(n, base_gs, inject_idx):
    """Feature: touchdown-point-detection, Property 9: Kinematic Gate Exclusion

    A smooth, plausible trajectory with an injected impossible groundspeed jump
    has the offending sample excluded, the count recorded, and every surviving
    consecutive transition physically plausible (Req 9.4).
    """
    inject = min(inject_idx, n - 1)
    dt = 4.0
    times = np.arange(n, dtype=float) * dt
    gs = np.full(n, base_gs, dtype=float)
    # Gentle, plausible deceleration elsewhere (~1 kt/s well under 1 g).
    gs = base_gs - 1.0 * np.arange(n, dtype=float)
    tracks = np.full(n, 90.0, dtype=float)
    # Inject an impossible instantaneous speed change at `inject` (huge jump).
    gs[inject] = gs[inject] + 200.0

    gates = _gates()
    result = apply_kinematic_gates(times, gs, tracks, gates)

    # The injected sample is excluded and counted.
    assert inject in result.excluded_indices.tolist()
    assert result.excluded_count >= 1
    assert times[inject] in result.excluded_timestamps
    assert sum(result.counts_by_gate.values()) == result.excluded_count

    # Survivors contain only plausible longitudinal transitions.
    kept = result.kept_indices
    kt = times[kept]
    kgs = gs[kept]
    g = STANDARD_GRAVITY_MPS2
    for j in range(1, kept.size):
        ddt = kt[j] - kt[j - 1]
        a_long = abs(kgs[j] - kgs[j - 1]) * KNOTS_TO_MPS / ddt
        assert a_long <= gates.max_longitudinal_accel_g * g + 1e-9


@pytest.mark.unit
def test_two_g_sample_excluded_with_timestamp():
    """A clearly impossible ~2 g longitudinal sample is excluded and logged."""
    # dt = 4 s; a 2 g change needs |dv| = 2*9.80665*4 = 78.45 m/s = 152.5 kt.
    times = np.array([0.0, 4.0, 8.0])
    gs = np.array([140.0, 140.0 - 160.0, 0.0])  # impossible 160 kt drop in 4 s
    # Clamp to a plausible-but-still-2g construction: use a big jump up instead.
    gs = np.array([140.0, 300.0, 140.0])  # +160 kt in 4 s ~ 2.1 g at idx 1
    tracks = np.array([90.0, 90.0, 90.0])
    result = apply_kinematic_gates(times, gs, tracks, _gates())
    assert 1 in result.excluded_indices.tolist()
    assert 4.0 in result.excluded_timestamps
    assert result.counts_by_gate["longitudinal_accel"] >= 1


@pytest.mark.unit
def test_turn_rate_gate_excludes_sharp_turn():
    """A turn faster than 6 deg/s (at low speed) is excluded by the turn-rate gate."""
    times = np.array([0.0, 1.0, 2.0])
    gs = np.array([20.0, 20.0, 20.0])  # low speed so lateral accel stays small
    tracks = np.array([0.0, 8.0, 16.0])  # 8 deg/s, over 6 deg/s but a_lat < 0.5 g
    result = apply_kinematic_gates(times, gs, tracks, _gates())
    assert result.excluded_count >= 1
    # Turn-rate (not lateral) should be the triggering gate at this low speed.
    assert result.counts_by_gate["turn_rate"] >= 1
    assert result.counts_by_gate["lateral_accel"] == 0


@pytest.mark.unit
def test_lateral_accel_gate_excludes_fast_turn():
    """A turn at high speed trips the lateral-acceleration gate (priority order)."""
    times = np.array([0.0, 1.0, 2.0])
    gs = np.array([200.0, 200.0, 200.0])
    tracks = np.array([0.0, 7.0, 14.0])  # 7 deg/s; a_lat = v*omega > 0.5 g here
    result = apply_kinematic_gates(times, gs, tracks, _gates())
    assert result.excluded_count >= 1
    assert result.counts_by_gate["lateral_accel"] >= 1


@pytest.mark.unit
def test_plausible_trajectory_excludes_nothing():
    """A gentle, plausible landing trajectory loses no samples."""
    times = np.arange(0.0, 40.0001, 4.0)
    gs = 150.0 - 3.0 * np.arange(times.size)  # ~3 kt/s, well under 1 g
    tracks = np.full(times.size, 250.0)
    result = apply_kinematic_gates(times, gs, tracks, _gates())
    assert result.excluded_count == 0
    assert result.kept_indices.size == times.size


# ---------------------------------------------------------------------------
# Property 13: Missing Vertical Rate Tolerance
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    n=st.integers(min_value=4, max_value=15),
    gs0=st.floats(min_value=120.0, max_value=160.0),
)
def test_missing_baro_vr_still_ingests(n, gs0):
    """Feature: touchdown-point-detection, Property 13: Missing Vertical Rate Tolerance

    A valid trajectory whose barometric vertical rate is entirely NaN still
    produces a usable (non-rejected) QA result, with a diagnostic noting baro VR
    is unavailable (Req 9.1).
    """
    dt = 4.0
    rows = []
    for i in range(n):
        t = i * dt
        on_ground = i >= n - 2
        rows.append(
            FR24Record(
                "VR", float(t), 33.9 + 0.0001 * i, -118.4, 300.0 - 5.0 * t,
                "geometric", max(gs0 - 3.0 * i, 60.0), 250.0, on_ground,
                vertical_rate_ftmin=None,  # entirely missing
            )
        )
    rec = parse_fr24_records(rows, runway=_runway(), aircraft_type="B738")
    assert np.all(np.isnan(rec.baro_vertical_rates))

    # Window centered on the on-ground transition; wide gates so it passes.
    result = run_qa(rec, _gates(min_samples_in_window=1))

    assert result.status == "ok"
    assert result.reason_code is None
    assert result.diagnostics.baro_vertical_rate_unavailable is True
    assert "baro_vertical_rate" in result.diagnostics.unavailable_signals


# ---------------------------------------------------------------------------
# Sufficiency gate: known-answer reason codes (Req 9.5 / 1.6)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_gap_spanning_touchdown_rejects():
    """A 16 s gap straddling the touchdown center -> GAP_SPANS_TOUCHDOWN."""
    # Samples at 0,4,8 then jump to 24,28,32: a 16 s gap over center=16.
    valid_times = np.array([0.0, 4.0, 8.0, 24.0, 28.0, 32.0])
    window = TouchdownWindow.from_center(16.0, 30.0)
    res = evaluate_sufficiency(
        valid_times=valid_times,
        n_excluded_in_window=0,
        groundspeed_present=True,
        window=window,
        gates=_gates(),
    )
    assert res.ok is False
    assert res.reason_code is FailureReason.GAP_SPANS_TOUCHDOWN
    assert res.max_gap_spanning_s == pytest.approx(16.0)


@pytest.mark.unit
def test_excessive_exclusions_rejects():
    """More than 50% of in-window samples excluded -> EXCESSIVE_EXCLUSIONS."""
    valid_times = np.array([10.0, 14.0])  # 2 valid in window
    window = TouchdownWindow.from_center(16.0, 30.0)
    res = evaluate_sufficiency(
        valid_times=valid_times,
        n_excluded_in_window=3,  # 3 excluded / (2+3)=0.6 > 0.5
        groundspeed_present=True,
        window=window,
        gates=_gates(min_samples_in_window=1),
    )
    assert res.ok is False
    assert res.reason_code is FailureReason.EXCESSIVE_EXCLUSIONS


@pytest.mark.unit
def test_insufficient_samples_rejects():
    """Fewer than min_samples_in_window valid in-window samples -> INSUFFICIENT_SAMPLES."""
    valid_times = np.array([15.0, 17.0])  # only 2 in window, need 3
    window = TouchdownWindow.from_center(16.0, 30.0)
    res = evaluate_sufficiency(
        valid_times=valid_times,
        n_excluded_in_window=0,
        groundspeed_present=True,
        window=window,
        gates=_gates(),
    )
    assert res.ok is False
    assert res.reason_code is FailureReason.INSUFFICIENT_SAMPLES


@pytest.mark.unit
def test_no_groundspeed_rejects():
    """Entirely missing groundspeed -> NO_GROUNDSPEED (highest precedence)."""
    window = TouchdownWindow.from_center(16.0, 30.0)
    res = evaluate_sufficiency(
        valid_times=np.array([15.0, 16.0, 17.0]),
        n_excluded_in_window=0,
        groundspeed_present=False,
        window=window,
        gates=_gates(),
    )
    assert res.ok is False
    assert res.reason_code is FailureReason.NO_GROUNDSPEED


@pytest.mark.unit
def test_run_qa_no_groundspeed_via_flight_record():
    """run_qa flags NO_GROUNDSPEED when the flight has no groundspeed at all."""
    rows = [
        FR24Record("NG", float(t), 33.9, -118.4, 300.0 - 10.0 * t, "geometric",
                   float("nan"), 250.0, t >= 8)
        for t in range(0, 16, 4)
    ]
    rec = parse_fr24_records(rows, runway=_runway(), aircraft_type="B738")
    result = run_qa(rec, _gates())
    assert result.status == "no-estimate"
    assert result.reason_code is FailureReason.NO_GROUNDSPEED


@pytest.mark.unit
def test_run_qa_dedup_and_gate_clean_record():
    """run_qa deduplicates and gates, returning an aligned cleaned co-timed record."""
    rows = [
        FR24Record("CL", 0.0, 33.90, -118.40, 300.0, "geometric", 150.0, 250.0, False),
        FR24Record("CL", 0.05, 33.90, -118.40, 300.0, "geometric", 150.0, 250.0, False),  # dup
        FR24Record("CL", 4.0, 33.901, -118.40, 280.0, "geometric", 147.0, 250.0, False),
        FR24Record("CL", 8.0, 33.902, -118.40, 260.0, "geometric", 144.0, 250.0, True),
        FR24Record("CL", 12.0, 33.903, -118.40, 240.0, "geometric", 141.0, 250.0, True),
    ]
    rec = parse_fr24_records(rows, runway=_runway(), aircraft_type="B738")
    result = run_qa(rec, _gates(min_samples_in_window=1))
    # One duplicate removed; co-timed arrays stay equal length and aligned.
    assert result.diagnostics.n_duplicates_removed == 1
    assert result.cleaned.position_times.size == result.cleaned.velocity_times.size
    assert np.array_equal(
        result.cleaned.position_times, result.cleaned.velocity_times
    )
    assert result.status == "ok"


# ---------------------------------------------------------------------------
# Constant
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_standard_gravity_constant():
    """The standard-gravity constant is exactly 9.80665 m/s^2."""
    assert STANDARD_GRAVITY_MPS2 == 9.80665
