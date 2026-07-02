"""Tests for the first-milestone validation report (Task 24).

Covers the assembly module :mod:`tdz.validation.milestone_report`, which wires
the stage-1-3 runner, the grouped-split held-out slice, the stratified
distance/time metrics (with the naive-baseline comparison, Req 12.8), the
cadence-limited error-floor characterization (Req 13.0), and the batch
provenance stamp into a single descriptive report.

The report is advisory -- it must never hard-fail on a below-target slice, and
the provisional targets stay unratified. The integration test runs the whole
report end-to-end on a synthetic corpus and asserts the four required pieces are
present (distance-error distribution, system-vs-baseline comparison, cadence
floor, provenance); the unit tests pin the pure "room to improve" arithmetic.

The synthetic corpus reuses ``synthetic_landing`` from
``test_physics_estimators`` (touchdown at the runway threshold, so the
along-runway truth distance is ~0 m) and QAR truth records carrying distinct
tail numbers so the tail-grouped split yields a non-empty held-out slice.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tdz.config.loader import load_config
from tdz.config.schema import TDZConfig
from tdz.models import QARTruthRecord
from tdz.reproducibility import BatchProvenance
from tdz.tests.test_physics_estimators import synthetic_landing
from tdz.validation import (
    MilestoneReport,
    RoomToImprove,
    build_milestone_report,
    compute_room_to_improve,
)
from tdz.validation.coverage import ErrorFloorReport
from tdz.validation.metrics import StratifiedMetricsReport

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "tdz_config.yaml"

# The synthetic-landing runway threshold (see synthetic_landing / _runway): the
# truth touchdown is placed here, so the along-runway truth distance is ~0 m.
_THRESHOLD_LAT = 33.94
_THRESHOLD_LON = -118.40


@pytest.fixture(scope="module")
def config() -> TDZConfig:
    """The canonical repo configuration (physics + change-point estimators)."""
    return load_config(str(_CONFIG_PATH))


# ---------------------------------------------------------------------------
# Synthetic corpus builders
# ---------------------------------------------------------------------------


def _truth(
    flight_id: str,
    *,
    t_td: float = 200.0,
    tail_number: str,
    aircraft_type: str = "B738",
    airport_id: str = "KLAX",
    runway_id: str = "25L",
) -> QARTruthRecord:
    """A QAR truth record whose touchdown is the runway threshold (truth ~0 m)."""
    return QARTruthRecord(
        flight_id=flight_id,
        touchdown_time_qar=t_td,
        touchdown_lat=_THRESHOLD_LAT,
        touchdown_lon=_THRESHOLD_LON,
        clock_offset_estimate=0.0,
        clock_offset_quality="good",
        aircraft_type=aircraft_type,
        runway_id=runway_id,
        airport_id=airport_id,
        tail_number=tail_number,
    )


def _synthetic_corpus(
    n: int = 24,
) -> tuple[list, list[QARTruthRecord]]:
    """Build ``n`` synthetic landings with distinct tails + their QAR truth.

    Distinct tail numbers guarantee the tail-grouped split can populate a
    non-empty held-out test partition. Phases/speeds are varied so the flights
    are not degenerate copies.
    """
    rng = np.random.default_rng(20240624)
    flights = []
    truths = []
    for i in range(n):
        t_td = 200.0
        phase = float(rng.uniform(0.0, 4.0))
        v = float(rng.uniform(60.0, 72.0))
        flight_id = f"MS{i:03d}"
        flights.append(
            synthetic_landing(
                dt=float(rng.uniform(4.0, 5.0)),
                t_td=t_td,
                v_td_mps=v,
                phase_s=phase,
                flight_id=flight_id,
            )
        )
        truths.append(_truth(flight_id, t_td=t_td, tail_number=f"N{i:04d}"))
    return flights, truths


# ===========================================================================
# Pure helper: room-to-improve arithmetic
# ===========================================================================


@pytest.mark.unit
def test_room_to_improve_headroom_above_floor():
    """Observed above the floor leaves positive headroom and a ratio > 1.

    Validates: Requirements 13.0
    """
    room = compute_room_to_improve(
        observed_rmse_ft=300.0, cadence_floor_ft=200.0, baseline_rmse_ft=1000.0
    )
    assert isinstance(room, RoomToImprove)
    assert room.headroom_above_floor_ft == pytest.approx(100.0)
    assert room.rmse_vs_floor_ratio == pytest.approx(1.5)
    assert room.improvement_over_baseline_pct == pytest.approx(70.0)
    assert room.at_floor is False


@pytest.mark.unit
def test_room_to_improve_at_or_below_floor_has_no_headroom():
    """Observed at/below the floor clamps headroom to zero and flags at_floor.

    Validates: Requirements 13.0
    """
    room = compute_room_to_improve(
        observed_rmse_ft=150.0, cadence_floor_ft=200.0, baseline_rmse_ft=800.0
    )
    assert room.headroom_above_floor_ft == 0.0
    assert room.at_floor is True
    assert room.rmse_vs_floor_ratio == pytest.approx(0.75)


@pytest.mark.unit
def test_room_to_improve_handles_nan_and_zero_gracefully():
    """NaN / zero inputs propagate to NaN derived quantities, never raising.

    Validates: Requirements 13.0
    """
    room = compute_room_to_improve(
        observed_rmse_ft=float("nan"), cadence_floor_ft=200.0, baseline_rmse_ft=0.0
    )
    assert math.isnan(room.headroom_above_floor_ft)
    assert math.isnan(room.rmse_vs_floor_ratio)
    assert math.isnan(room.improvement_over_baseline_pct)
    assert room.at_floor is False


# ===========================================================================
# Integration: end-to-end milestone report on a synthetic corpus
# ===========================================================================


@pytest.mark.integration
def test_milestone_report_end_to_end(config: TDZConfig):
    """The report runs stages 1-3 on the held-out slice and carries all pieces.

    Asserts the four required deliverables are present -- a distance-error
    distribution, a system-vs-baseline comparison (Req 12.8), a cadence-floor
    characterization (Req 13.0), and batch provenance -- and that the run does
    not hard-fail on the held-out slice.

    Validates: Requirements 13.0, 12.8
    """
    flights, truths = _synthetic_corpus(n=24)
    report = build_milestone_report(
        flights, truths, config, code_commit="test-commit"
    )

    assert isinstance(report, MilestoneReport)
    # A non-empty held-out slice was drawn and evaluated.
    assert report.slice_selector == "test"
    assert report.split_group_key == "tail"
    assert report.n_slice_flights > 0
    assert report.n_evaluated > 0
    assert report.n_evaluated + report.n_no_estimate == report.n_slice_flights

    # (1) Distance-error distribution.
    assert isinstance(report.metrics, StratifiedMetricsReport)
    overall = report.metrics.overall
    assert overall.n_flights == report.n_evaluated
    assert math.isfinite(overall.distance_rmse_ft)
    assert math.isfinite(overall.distance_median_abs_error_ft)
    assert math.isfinite(overall.distance_p95_abs_error_ft)

    # (2) System-vs-baseline comparison (Req 12.8).
    assert math.isfinite(overall.baseline_rmse_ft)
    # On the clean synthetic set the physics baseline beats the naive strawman.
    assert overall.distance_rmse_ft < overall.baseline_rmse_ft
    assert report.room_to_improve.improvement_over_baseline_pct > 0.0

    # (3) Cadence-limited error-floor characterization (Req 13.0).
    assert isinstance(report.error_floor, ErrorFloorReport)
    assert report.error_floor.cadence_s == pytest.approx(
        config.uncertainty.nominal_cadence_s
    )
    assert math.isfinite(report.error_floor.floor_ft)
    assert report.error_floor.floor_ft > 0.0

    # (4) Batch provenance (Req 15.3).
    assert isinstance(report.provenance, BatchProvenance)
    assert report.provenance.code_commit == "test-commit"
    assert report.provenance.library_versions["numpy"] == np.__version__

    # Advisory-only: targets are never ratified by this milestone, and it does
    # not raise on below-target strata.
    assert report.targets_ratified is False
    assert isinstance(report.below_target_flags, tuple)

    # The summary dict surfaces the headline numbers.
    summary = report.to_summary_dict()
    assert summary["n_evaluated"] == report.n_evaluated
    assert summary["distance_rmse_ft"] == pytest.approx(overall.distance_rmse_ft)
    assert summary["cadence_floor_ft"] == pytest.approx(report.error_floor.floor_ft)
    assert "provenance" in summary


@pytest.mark.integration
def test_milestone_report_room_reflects_floor_and_baseline(config: TDZConfig):
    """The room-to-improve fields are consistent with the metrics and floor.

    On a clean synthetic corpus the observed error sits at/near the cadence
    floor (little geometric headroom) yet far below the naive baseline.

    Validates: Requirements 13.0, 12.8
    """
    flights, truths = _synthetic_corpus(n=24)
    report = build_milestone_report(flights, truths, config, code_commit="c")

    room = report.room_to_improve
    assert room.observed_rmse_ft == pytest.approx(
        report.metrics.overall.distance_rmse_ft
    )
    assert room.cadence_floor_ft == pytest.approx(report.error_floor.floor_ft)
    assert room.baseline_rmse_ft == pytest.approx(
        report.metrics.overall.baseline_rmse_ft
    )
    # Headroom is the clamped gap to the floor.
    expected_headroom = max(room.observed_rmse_ft - room.cadence_floor_ft, 0.0)
    assert room.headroom_above_floor_ft == pytest.approx(expected_headroom)
    # Clean synthetic errors are small: at or below the cadence floor.
    assert room.at_floor is True


@pytest.mark.integration
def test_milestone_report_empty_slice_does_not_hard_fail(config: TDZConfig):
    """A corpus with no matching held-out flights yields an empty, safe report.

    The report must remain well-defined (no raise) even when the held-out slice
    is empty -- the metrics are NaN/empty rather than an error.

    Validates: Requirements 13.0
    """
    flights, truths = _synthetic_corpus(n=6)
    # Restrict truths to a set of flights, but pass NO flights to evaluate.
    report = build_milestone_report([], truths, config, code_commit="c")

    assert report.n_slice_flights == 0
    assert report.n_evaluated == 0
    assert report.metrics.overall.n_flights == 0
    assert math.isnan(report.metrics.overall.distance_rmse_ft)
    assert report.below_target_flags == ()
    assert report.room_to_improve.at_floor is False
