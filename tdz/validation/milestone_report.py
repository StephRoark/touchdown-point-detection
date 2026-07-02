"""First-milestone validation report (Task 24).

This module is the final **assembly** step of the spec: it wires together the
pieces already built in earlier tasks into a single descriptive report over a
held-out QAR slice. It invents no new estimation logic -- it reuses:

* the stage-1-3 runner (:func:`tdz.pipeline.run_stage123`) and the naive
  first-on-ground baseline (:func:`tdz.pipeline.naive_baseline_distance`,
  surfaced on every :class:`~tdz.pipeline.StageRunResult`);
* the validation harness splits (:func:`tdz.validation.make_validation_splits`)
  to carve out the leakage-controlled held-out test slice;
* the stratified distance/time metrics
  (:func:`tdz.validation.compute_stratified_metrics`), which already place the
  system stats side-by-side with the naive baseline (Req 12.8);
* the cadence-limited error-floor characterization
  (:func:`tdz.validation.characterize_error_floor`; Req 13.0); and
* the batch provenance stamp (:func:`tdz.reproducibility.resolve_batch_provenance`).

Purpose (Req 13.0, 12.8)
------------------------
The provisional accuracy targets in Requirement 13 are **reporting targets, not
pass/fail gates**, until this milestone empirically characterizes the
irreducible cadence-limited error floor. This report is therefore
**descriptive/advisory and never hard-fails**: it observes where the current
physics + change-point baseline sits relative to (a) the cadence-limited floor
and (b) the naive strawman, and surfaces *how much room the learned models have
to add* before the provisional targets can be ratified. Below-target strata are
*flagged*, never raised.

Bridging stage results to the metrics harness
----------------------------------------------
:func:`tdz.pipeline.run_stage123` produces a provisional
:class:`~tdz.pipeline.StageRunResult` in SI (metres), whereas the metrics
harness consumes :class:`~tdz.validation.FlightEvaluation` objects carrying a
presentation-unit :class:`~tdz.models.TouchdownResult`. This module bridges the
two at the same single SI->feet boundary the rest of the pipeline uses
(:data:`~tdz.uncertainty.M_TO_FT`). The distance truth stays clock-independent
(projected from the QAR touchdown lat/long; Req 12.10), so no clock alignment is
required for this report.

Units: distances in feet (via :data:`~tdz.uncertainty.M_TO_FT`), times in
seconds, speeds in knots. The cadence floor reuses
``config.uncertainty.nominal_cadence_s``; split/metric knobs come from
``config.validation``. No estimation-affecting numeric literals appear here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Optional, Sequence

import numpy as np

from tdz.config.schema import ProvisionalAccuracyTargets, TDZConfig
from tdz.models import FlightRecord, QARTruthRecord, TouchdownResult
from tdz.pipeline import StageRunResult, run_stage123
from tdz.reproducibility import BatchProvenance, resolve_batch_provenance
from tdz.timebase.interpolation import KNOTS_TO_MPS, interpolate_groundspeed_at
from tdz.uncertainty import M_TO_FT, gaussian_multiplier
from tdz.validation.coverage import (
    BelowTargetFlag,
    ErrorFloorReport,
    characterize_error_floor,
    flag_below_target,
)
from tdz.validation.metrics import (
    NO_ESTIMATE,
    FlightEvaluation,
    StratifiedMetricsReport,
    compute_stratified_metrics,
)
from tdz.validation.splits import ValidationSplits, make_validation_splits

__all__ = [
    "CONFIDENCE_NORMAL",
    "SLICE_SELECTORS",
    "RoomToImprove",
    "MilestoneReport",
    "compute_room_to_improve",
    "build_milestone_report",
]

#: Confidence class stamped on a bridged result that produced a usable estimate.
#: Any value other than :data:`~tdz.validation.metrics.NO_ESTIMATE` is included
#: by the metrics harness; ``"normal"`` matches the pipeline's own vocabulary.
CONFIDENCE_NORMAL: Final[str] = "normal"

#: The primary-split partitions that may be used as the held-out slice, plus the
#: two generalization-stress test partitions. ``"test"`` is the headline
#: held-out slice (Req 12.2/13.1).
SLICE_SELECTORS: Final[tuple[str, ...]] = (
    "test",
    "calibration",
    "train",
    "held_out_airport",
    "held_out_runway",
)


# ---------------------------------------------------------------------------
# "Room to improve" -- pure, descriptive headroom computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoomToImprove:
    """How much room the learned models have to add over this baseline (Req 13.0).

    All feet. The comparison is purely descriptive -- it never gates anything.

    Attributes
    ----------
    observed_rmse_ft:
        Overall along-runway distance RMSE of the current (physics +
        change-point) baseline on the held-out slice.
    cadence_floor_ft:
        The representative cadence-limited error floor (from
        :class:`~tdz.validation.ErrorFloorReport`): the irreducible resolution
        imposed by the ~4-5 s update interval at approach groundspeed. No
        estimator can drive the RMSE below this from sample geometry alone.
    baseline_rmse_ft:
        The naive first-on-ground baseline RMSE on the same slice.
    headroom_above_floor_ft:
        ``max(observed - floor, 0)`` -- the distance the RMSE could still be
        driven down toward the irreducible floor. This is the room a better
        (e.g. learned) estimator could plausibly reclaim; ``0`` means the
        baseline already sits at/below the floor and there is nothing left to
        gain from sample geometry.
    rmse_vs_floor_ratio:
        ``observed / floor`` -- how many times the floor the observed RMSE is.
    improvement_over_baseline_pct:
        ``(baseline - observed) / baseline * 100`` -- the RMSE reduction the
        current system already achieves over the naive strawman (Req 12.8/13.4).
    at_floor:
        ``True`` when the observed RMSE is already at or below the floor (no
        residual headroom to reclaim from geometry).
    """

    observed_rmse_ft: float
    cadence_floor_ft: float
    baseline_rmse_ft: float
    headroom_above_floor_ft: float
    rmse_vs_floor_ratio: float
    improvement_over_baseline_pct: float
    at_floor: bool


def compute_room_to_improve(
    observed_rmse_ft: float,
    cadence_floor_ft: float,
    baseline_rmse_ft: float,
) -> RoomToImprove:
    """Compute the descriptive headroom of the current baseline (Req 13.0).

    Pure function of three RMSE-scale quantities (all feet). NaN inputs
    propagate to NaN derived quantities rather than raising, so the report stays
    well-defined even on an empty or degenerate slice. ``at_floor`` is ``False``
    whenever the comparison is undefined (a non-finite observed RMSE or floor).
    """
    observed = float(observed_rmse_ft)
    floor = float(cadence_floor_ft)
    baseline = float(baseline_rmse_ft)

    if np.isfinite(observed) and np.isfinite(floor):
        headroom = max(observed - floor, 0.0)
        at_floor = observed <= floor
    else:
        headroom = float("nan")
        at_floor = False

    if np.isfinite(observed) and np.isfinite(floor) and floor > 0.0:
        ratio = observed / floor
    else:
        ratio = float("nan")

    if np.isfinite(observed) and np.isfinite(baseline) and baseline > 0.0:
        improvement = (baseline - observed) / baseline * 100.0
    else:
        improvement = float("nan")

    return RoomToImprove(
        observed_rmse_ft=observed,
        cadence_floor_ft=floor,
        baseline_rmse_ft=baseline,
        headroom_above_floor_ft=headroom,
        rmse_vs_floor_ratio=ratio,
        improvement_over_baseline_pct=improvement,
        at_floor=at_floor,
    )


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MilestoneReport:
    """The first-milestone validation report (Task 24; Req 13.0, 12.8).

    A structured, **advisory** snapshot of the current stage-1-3 baseline on a
    held-out QAR slice. ``targets_ratified`` is always ``False``: the point of
    this milestone is to characterize the floor so the provisional targets *can*
    be ratified later -- it does not itself ratify or gate them.

    Attributes
    ----------
    slice_selector, split_group_key:
        Which partition was used as the held-out slice, and the grouping rule
        that produced it (e.g. ``"tail"``; Req 12.2).
    n_slice_flights:
        Flights in the held-out slice that carried a QAR truth and were run.
    n_evaluated:
        Flights that produced a usable estimate (contributed to the metrics).
    n_no_estimate:
        Flights in the slice that produced no estimate (go-around / QA reject /
        all estimators failed).
    metrics:
        The stratified distance/time metrics, already carrying the system-vs-
        naive-baseline side-by-side comparison (Req 12.8).
    error_floor:
        The cadence-limited error-floor characterization (Req 13.0).
    room_to_improve:
        The descriptive headroom of the current baseline against the floor and
        the naive strawman.
    below_target_flags:
        Advisory below-target stratum flags against the provisional targets
        (Req 13.5). Empty on a small slice (they require >= 200 flights/stratum).
        Never a failure.
    provisional_targets:
        The provisional accuracy targets, echoed for reference. NOT gates.
    provenance:
        The batch provenance stamp (Req 15.3).
    targets_ratified:
        Always ``False`` -- see the class docstring.
    """

    slice_selector: str
    split_group_key: str
    n_slice_flights: int
    n_evaluated: int
    n_no_estimate: int
    metrics: StratifiedMetricsReport
    error_floor: ErrorFloorReport
    room_to_improve: RoomToImprove
    below_target_flags: tuple[BelowTargetFlag, ...]
    provisional_targets: ProvisionalAccuracyTargets
    provenance: BatchProvenance
    targets_ratified: bool = False

    def to_summary_dict(self) -> dict:
        """A flat, human-readable summary of the headline numbers.

        Intended for logging / serialising the report envelope; the full
        distributions live on :attr:`metrics` and :attr:`error_floor`.
        """
        overall = self.metrics.overall
        return {
            "slice_selector": self.slice_selector,
            "split_group_key": self.split_group_key,
            "n_slice_flights": self.n_slice_flights,
            "n_evaluated": self.n_evaluated,
            "n_no_estimate": self.n_no_estimate,
            "targets_ratified": self.targets_ratified,
            "distance_rmse_ft": overall.distance_rmse_ft,
            "distance_median_abs_error_ft": overall.distance_median_abs_error_ft,
            "distance_p95_abs_error_ft": overall.distance_p95_abs_error_ft,
            "distance_median_signed_error_ft": overall.distance_median_signed_error_ft,
            "baseline_rmse_ft": overall.baseline_rmse_ft,
            "improvement_over_baseline_pct": self.room_to_improve.improvement_over_baseline_pct,
            "cadence_floor_ft": self.error_floor.floor_ft,
            "cadence_s": self.error_floor.cadence_s,
            "fraction_within_floor": self.error_floor.fraction_within_floor,
            "headroom_above_floor_ft": self.room_to_improve.headroom_above_floor_ft,
            "rmse_vs_floor_ratio": self.room_to_improve.rmse_vs_floor_ratio,
            "at_floor": self.room_to_improve.at_floor,
            "n_below_target_flags": len(self.below_target_flags),
            "provenance": self.provenance.to_dict(),
        }


# ---------------------------------------------------------------------------
# Stage-result -> evaluation bridge
# ---------------------------------------------------------------------------


def _touchdown_groundspeed_kt(flight: FlightRecord, t_td: Optional[float]) -> float:
    """Interpolate the groundspeed (knots) at the combined touchdown time.

    Reuses :func:`tdz.timebase.interpolation.interpolate_groundspeed_at` on the
    flight's velocity timebase; returns ``nan`` when no touchdown time is
    available or interpolation cannot be evaluated.
    """
    if t_td is None or not np.isfinite(t_td):
        return float("nan")
    try:
        return float(
            interpolate_groundspeed_at(
                flight.velocity_times, flight.groundspeeds, float(t_td)
            )
        )
    except (ValueError, IndexError):
        return float("nan")


def _stage_result_to_evaluation(
    result: StageRunResult,
    flight: FlightRecord,
    truth: QARTruthRecord,
    provenance: BatchProvenance,
    coverage_target: float,
) -> FlightEvaluation:
    """Bridge a :class:`~tdz.pipeline.StageRunResult` to a :class:`FlightEvaluation`.

    Converts the provisional SI combined distance to feet at the single SI->feet
    boundary and packages it as a :class:`~tdz.models.TouchdownResult` the
    metrics harness understands. A no-touchdown / no-combined-estimate result is
    stamped :data:`~tdz.validation.metrics.NO_ESTIMATE` so it is excluded from
    the metrics (its numeric fields are ``NaN``). The provisional distance CI is
    derived from the combiner's ``sigma_t`` projected through the interpolated
    groundspeed -- a first-milestone stand-in for the calibrated Task-19/20
    interval, adequate for a descriptive report.
    """
    record_prov = provenance.to_record_provenance()
    reason_code = result.reason_code.value if result.reason_code is not None else None
    trajectory_type = str(result.diagnostics.get("trajectory_type", "completed-landing"))
    contributing = list(result.combined.contributing)
    excluded = (
        list(result.gating.excluded_estimators) if result.gating is not None else []
    )

    usable = (
        not result.no_touchdown
        and result.combined.ok
        and result.combined_distance_m is not None
    )

    if not usable:
        nan = float("nan")
        no_estimate_result = TouchdownResult(
            flight_id=result.flight_id,
            aircraft_type=flight.aircraft_type,
            ads_b_source=result.ads_b_source,
            touchdown_time=nan,
            along_runway_distance_ft=nan,
            lateral_offset_ft=nan,
            groundspeed_at_touchdown_kt=nan,
            time_ci_90_lower=nan,
            time_ci_90_upper=nan,
            distance_ci_90_lower_ft=nan,
            distance_ci_90_upper_ft=nan,
            speed_ci_90_lower_kt=nan,
            speed_ci_90_upper_kt=nan,
            trajectory_type=trajectory_type,
            confidence=NO_ESTIMATE,
            reason_code=reason_code,
            contributing_estimators=contributing,
            excluded_estimators=excluded,
            physics_anchor_t_td=result.combined_t_td,
            physics_anchor_diagnostics=None,
            lever_arm_used=None,
            lever_arm_missing=False,
            assumed_touchdown_pitch_deg=nan,
            geometric_altitude_available=True,
            runway_elevation_datum=str(flight.runway.elevation_datum),
            suspected_wrong_runway=False,
            out_of_bounds=False,
            data_version=record_prov.data_version,
            code_commit=record_prov.code_commit,
            config_hash=record_prov.config_hash,
            model_artifact_hash=record_prov.model_artifact_hash,
        )
        return FlightEvaluation(
            result=no_estimate_result,
            truth=truth,
            runway=flight.runway,
            baseline_distance_m=None,
        )

    distance_ft = float(result.combined_distance_m) * M_TO_FT
    lateral_ft = (
        float(result.lateral_offset_m) * M_TO_FT
        if result.lateral_offset_m is not None
        else float("nan")
    )
    t_td = float(result.combined_t_td)
    gs_kt = _touchdown_groundspeed_kt(flight, t_td)

    # Provisional intervals: project the combiner's time sigma through the
    # groundspeed for the distance CI (first-milestone stand-in for the
    # calibrated Task-19/20 interval). Purely descriptive here.
    z = gaussian_multiplier(coverage_target)
    sigma_t = float(result.combined.sigma_t)
    v_mps = gs_kt * KNOTS_TO_MPS if np.isfinite(gs_kt) else float("nan")
    if np.isfinite(sigma_t) and np.isfinite(v_mps):
        half_distance_ft = z * sigma_t * v_mps * M_TO_FT
        distance_ci_lower_ft = distance_ft - half_distance_ft
        distance_ci_upper_ft = distance_ft + half_distance_ft
    else:
        distance_ci_lower_ft = float("nan")
        distance_ci_upper_ft = float("nan")

    if np.isfinite(sigma_t):
        time_ci_lower = t_td - z * sigma_t
        time_ci_upper = t_td + z * sigma_t
    else:
        time_ci_lower = float("nan")
        time_ci_upper = float("nan")

    touchdown_result = TouchdownResult(
        flight_id=result.flight_id,
        aircraft_type=flight.aircraft_type,
        ads_b_source=result.ads_b_source,
        touchdown_time=t_td,
        along_runway_distance_ft=distance_ft,
        lateral_offset_ft=lateral_ft,
        groundspeed_at_touchdown_kt=gs_kt,
        time_ci_90_lower=time_ci_lower,
        time_ci_90_upper=time_ci_upper,
        distance_ci_90_lower_ft=distance_ci_lower_ft,
        distance_ci_90_upper_ft=distance_ci_upper_ft,
        speed_ci_90_lower_kt=gs_kt,
        speed_ci_90_upper_kt=gs_kt,
        trajectory_type=trajectory_type,
        confidence=CONFIDENCE_NORMAL,
        reason_code=reason_code,
        contributing_estimators=contributing,
        excluded_estimators=excluded,
        physics_anchor_t_td=result.combined_t_td,
        physics_anchor_diagnostics=None,
        lever_arm_used=None,
        lever_arm_missing=False,
        assumed_touchdown_pitch_deg=float("nan"),
        geometric_altitude_available=True,
        runway_elevation_datum=str(flight.runway.elevation_datum),
        suspected_wrong_runway=False,
        out_of_bounds=False,
        data_version=record_prov.data_version,
        code_commit=record_prov.code_commit,
        config_hash=record_prov.config_hash,
        model_artifact_hash=record_prov.model_artifact_hash,
    )

    baseline_distance_m = (
        result.naive_baseline.distance_m if result.naive_baseline.available else None
    )
    return FlightEvaluation(
        result=touchdown_result,
        truth=truth,
        runway=flight.runway,
        baseline_distance_m=baseline_distance_m,
    )


# ---------------------------------------------------------------------------
# Slice selection
# ---------------------------------------------------------------------------


def _slice_ids(splits: ValidationSplits, selector: str) -> tuple[frozenset[str], str]:
    """Resolve the held-out slice's flight ids and the grouping rule used."""
    if selector == "test":
        return frozenset(splits.primary.test), splits.primary.group_key
    if selector == "calibration":
        return frozenset(splits.primary.calibration), splits.primary.group_key
    if selector == "train":
        return frozenset(splits.primary.train), splits.primary.group_key
    if selector == "held_out_airport":
        if splits.held_out_airport is None:
            return frozenset(), "airport"
        return frozenset(splits.held_out_airport.test), splits.held_out_airport.group_key
    if selector == "held_out_runway":
        if splits.held_out_runway is None:
            return frozenset(), "runway"
        return frozenset(splits.held_out_runway.test), splits.held_out_runway.group_key
    raise ValueError(
        f"unknown slice selector {selector!r}; expected one of {SLICE_SELECTORS}"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_milestone_report(
    flights: Sequence[FlightRecord],
    truths: Sequence[QARTruthRecord] | Mapping[str, QARTruthRecord],
    config: TDZConfig,
    *,
    slice_selector: str = "test",
    data_version: str = "unknown",
    code_commit: Optional[str] = None,
) -> MilestoneReport:
    """Assemble the first-milestone validation report (Task 24; Req 13.0, 12.8).

    Builds the leakage-controlled validation splits over the QAR truth corpus,
    runs stages 1-3 on the requested held-out slice, computes the stratified
    distance-error distribution with the system-vs-naive-baseline comparison
    (Req 12.8), characterizes the cadence-limited error floor (Req 13.0), and
    stamps the whole thing with batch provenance (Req 15.3).

    This is **descriptive and advisory**: the provisional accuracy targets are
    reporting targets, not gates, until this milestone characterizes the floor,
    so the function never raises on a below-target result -- below-target strata
    are merely flagged.

    Parameters
    ----------
    flights:
        The corpus flight records. Only those whose ``flight_id`` falls in the
        selected held-out slice (and that carry a QAR truth) are evaluated.
    truths:
        The QAR truth records for the corpus, as a sequence or a
        ``{flight_id: QARTruthRecord}`` mapping. The full corpus drives the
        grouped split; the held-out partition selects the evaluated flights.
    config:
        The resolved configuration. ``config.validation`` drives the splits and
        metric knobs; ``config.uncertainty.nominal_cadence_s`` drives the
        cadence floor; ``config.uncertainty.coverage_target`` sizes the
        provisional CIs used in the bridge.
    slice_selector:
        Which partition to use as the held-out slice (default ``"test"``; see
        :data:`SLICE_SELECTORS`).
    data_version, code_commit:
        Passed through to :func:`tdz.reproducibility.resolve_batch_provenance`
        (``code_commit`` may be supplied to stay hermetic in tests).

    Returns
    -------
    MilestoneReport
        The structured, provenance-stamped report.
    """
    truth_by_id: dict[str, QARTruthRecord] = (
        dict(truths)
        if isinstance(truths, Mapping)
        else {t.flight_id: t for t in truths}
    )

    splits = make_validation_splits(list(truth_by_id.values()), config)
    slice_ids, group_key = _slice_ids(splits, slice_selector)

    provenance = resolve_batch_provenance(
        config, data_version=data_version, code_commit=code_commit
    )
    coverage_target = float(config.uncertainty.coverage_target)

    evaluations: list[FlightEvaluation] = []
    n_slice_flights = 0
    n_no_estimate = 0
    for flight in flights:
        if flight.flight_id not in slice_ids:
            continue
        truth = truth_by_id.get(flight.flight_id)
        if truth is None:
            continue
        n_slice_flights += 1
        result = run_stage123(flight, config)
        evaluation = _stage_result_to_evaluation(
            result, flight, truth, provenance, coverage_target
        )
        if evaluation.result.confidence == NO_ESTIMATE:
            n_no_estimate += 1
        evaluations.append(evaluation)

    metrics = compute_stratified_metrics(evaluations, config.validation)
    error_floor = characterize_error_floor(
        evaluations, config.validation, config.uncertainty.nominal_cadence_s
    )
    below_target_flags = flag_below_target(metrics, config.validation)
    room = compute_room_to_improve(
        metrics.overall.distance_rmse_ft,
        error_floor.floor_ft,
        metrics.overall.baseline_rmse_ft,
    )

    return MilestoneReport(
        slice_selector=slice_selector,
        split_group_key=group_key,
        n_slice_flights=n_slice_flights,
        n_evaluated=int(metrics.overall.n_flights),
        n_no_estimate=n_no_estimate,
        metrics=metrics,
        error_floor=error_floor,
        room_to_improve=room,
        below_target_flags=below_target_flags,
        provisional_targets=config.validation.provisional_targets,
        provenance=provenance,
    )
