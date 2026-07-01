"""CI coverage assessment, cadence-limited error floor, and below-target flags (Task 22.4).

This module closes out the validation harness (parent Task 22) with three
reporting facilities, all computed on the leakage-controlled calibration split
(:attr:`~tdz.validation.GroupedSplit.calibration`) and all *report-only* -- none
of them raises or aborts:

1. **Empirical coverage assessment** (Req 4.3, 4.4). The reported 90% CIs are
   supposed to achieve empirical coverage in the ``[coverage_min, coverage_max]``
   band (default 85%-95%). We measure that coverage for **both** the time CI and
   the distance CI and classify each as *in-band*, *under* (< min, unsafe) or
   *over* (> max, uninformative). Distance coverage is assessed **without any
   clock-offset correction**: the distance truth is a pure geometric projection
   of the QAR touchdown lat/long (Req 12.10), so it is clock-independent by
   construction. Time coverage necessarily uses the clock-aligned QAR touchdown
   time and therefore excludes flights whose clock offset could not be reliably
   estimated (``clock_offset_quality == "failed"``), mirroring the time-error
   metric in :mod:`tdz.validation.metrics`.

   This *measures* coverage; it is complementary to the conformal calibrator in
   :mod:`tdz.uncertainty.conformal` (Task 19), which *sets* interval widths so
   that measured coverage lands in-band.

2. **Cadence-limited error floor** (Req 13.0, 13.1). ADS-B updates arrive only
   every ~4-5 s, so between two samples the along-runway position is known only
   to within the distance the aircraft travels in one update interval. At an
   approach groundspeed ``v`` and nominal cadence ``C`` that interpolation-limited
   resolution is ``v * C`` -- the irreducible floor below which no estimator can
   drive the distance error, regardless of method. :func:`cadence_limited_floor_ft`
   returns that floor in feet; :func:`characterize_error_floor` reports the
   observed error distribution against it.

3. **Below-target flagging** (Req 13.2-13.5). Per-stratum metrics are compared
   against the **provisional** accuracy targets (Req 13, criteria 1-3) and
   below-target strata are *flagged* -- but only for strata holding at least
   ``below_target_min_flights`` flights (default 200; distinct from the >=30
   reporting gate in Task 22.2). The targets are provisional until ratified
   against the floor above (Req 13.0), so this **never hard-fails**: it returns a
   tuple of flags for the report.

Units follow :mod:`tdz.validation.metrics`: distances in feet (via
:data:`~tdz.uncertainty.M_TO_FT`), times in seconds, speeds in knots. The
knots->feet-per-second conversion reuses the documented
:data:`~tdz.timebase.KNOTS_TO_MPS` constant, so no estimation-affecting numeric
literal appears here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from tdz.config.schema import ValidationConfig
from tdz.timebase import KNOTS_TO_MPS
from tdz.uncertainty import M_TO_FT
from tdz.validation.clock_alignment import QUALITY_FAILED
from tdz.validation.metrics import (
    NO_ESTIMATE,
    FlightEvaluation,
    StratifiedMetricsReport,
    compute_metrics,
)

__all__ = [
    "COVERAGE_IN_BAND",
    "COVERAGE_UNDER",
    "COVERAGE_OVER",
    "COVERAGE_UNDEFINED",
    "KNOTS_TO_FT_PER_S",
    "CoverageAssessment",
    "ErrorFloorReport",
    "BelowTargetFlag",
    "classify_coverage",
    "assess_coverage",
    "cadence_limited_floor_ft",
    "characterize_error_floor",
    "flag_below_target",
]

#: Empirical coverage falls inside the acceptance band ``[coverage_min, coverage_max]``.
COVERAGE_IN_BAND: str = "in-band"
#: Empirical coverage is below ``coverage_min`` -- intervals are too narrow (unsafe).
COVERAGE_UNDER: str = "under"
#: Empirical coverage is above ``coverage_max`` -- intervals are too wide (uninformative).
COVERAGE_OVER: str = "over"
#: No flights were available to assess coverage (coverage is undefined / NaN).
COVERAGE_UNDEFINED: str = "undefined"

#: Exact knots -> feet/second, composed from the two documented unit constants.
KNOTS_TO_FT_PER_S: float = KNOTS_TO_MPS * M_TO_FT

#: Below-target flagging is defined by Req 13.5 for ADS-B source and aircraft
#: type strata; these are the dimensions inspected by :func:`flag_below_target`.
_FLAG_DIMENSIONS: tuple[str, ...] = ("aircraft_type", "source")


# ---------------------------------------------------------------------------
# Coverage assessment (Req 4.3, 4.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoverageAssessment:
    """Empirical 90%-CI coverage for the time and distance intervals.

    ``distance_coverage`` is the fraction of flights whose clock-independent
    distance truth lies inside the reported distance CI (Req 4.4); ``time_coverage``
    is the analogous fraction for the time CI against the clock-aligned QAR
    touchdown time (Req 4.3). ``*_classification`` labels each as
    :data:`COVERAGE_IN_BAND` / :data:`COVERAGE_UNDER` / :data:`COVERAGE_OVER`
    (or :data:`COVERAGE_UNDEFINED` when the corresponding sample is empty).
    ``n_time`` may be smaller than ``n_distance`` because failed-clock flights
    are excluded from the time assessment only.
    """

    n_distance: int
    distance_coverage: float
    distance_classification: str
    n_time: int
    time_coverage: float
    time_classification: str
    coverage_min: float
    coverage_max: float


def classify_coverage(coverage: float, coverage_min: float, coverage_max: float) -> str:
    """Classify empirical ``coverage`` against the acceptance band.

    Returns :data:`COVERAGE_UNDEFINED` for a non-finite coverage (empty sample),
    :data:`COVERAGE_UNDER` when ``coverage < coverage_min`` (intervals too narrow,
    unsafe), :data:`COVERAGE_OVER` when ``coverage > coverage_max`` (intervals too
    wide, uninformative), and :data:`COVERAGE_IN_BAND` otherwise. The band is
    inclusive at both edges.
    """
    if not np.isfinite(coverage):
        return COVERAGE_UNDEFINED
    if coverage < coverage_min:
        return COVERAGE_UNDER
    if coverage > coverage_max:
        return COVERAGE_OVER
    return COVERAGE_IN_BAND


def _time_ci_coverage(evaluations: Sequence[FlightEvaluation]) -> tuple[int, float]:
    """Empirical time-CI coverage and its sample size (Req 4.3).

    Uses the clock-aligned QAR touchdown time as truth and excludes no-estimate
    flights and failed-clock flights (whose aligned time is untrustworthy),
    matching the time-error metric in :mod:`tdz.validation.metrics`.
    """
    inside = 0
    n = 0
    for ev in evaluations:
        if ev.result.confidence == NO_ESTIMATE:
            continue
        if ev.truth.clock_offset_quality == QUALITY_FAILED:
            continue
        lo = float(ev.result.time_ci_90_lower)
        hi = float(ev.result.time_ci_90_upper)
        truth_t = float(ev.truth.touchdown_time_qar)
        if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(truth_t)):
            continue
        n += 1
        if lo <= truth_t <= hi:
            inside += 1
    if n == 0:
        return 0, float("nan")
    return n, inside / n


def assess_coverage(
    evaluations: Sequence[FlightEvaluation],
    config: ValidationConfig,
) -> CoverageAssessment:
    """Assess empirical 90%-CI coverage on the calibration split (Req 4.3, 4.4).

    ``evaluations`` are the paired system/truth records for the flights in the
    calibration partition (:attr:`~tdz.validation.GroupedSplit.calibration`).
    Distance coverage reuses :func:`~tdz.validation.metrics.compute_metrics`
    (its ``ci_90_coverage`` is the clock-independent distance-CI coverage, Req
    4.4, 12.10); time coverage is computed here against the clock-aligned QAR
    touchdown time (Req 4.3). Each coverage is classified against
    ``config.coverage_min`` / ``config.coverage_max``.
    """
    distance_metrics = compute_metrics(evaluations)
    n_distance = int(distance_metrics.n_flights)
    distance_coverage = float(distance_metrics.ci_90_coverage)

    n_time, time_coverage = _time_ci_coverage(evaluations)

    cmin = float(config.coverage_min)
    cmax = float(config.coverage_max)

    return CoverageAssessment(
        n_distance=n_distance,
        distance_coverage=distance_coverage,
        distance_classification=classify_coverage(distance_coverage, cmin, cmax),
        n_time=n_time,
        time_coverage=time_coverage,
        time_classification=classify_coverage(time_coverage, cmin, cmax),
        coverage_min=cmin,
        coverage_max=cmax,
    )


# ---------------------------------------------------------------------------
# Cadence-limited error floor (Req 13.0, 13.1)
# ---------------------------------------------------------------------------


def cadence_limited_floor_ft(groundspeed_kt: float, cadence_s: float) -> float:
    """Interpolation-limited distance resolution in feet imposed by the cadence.

    At approach groundspeed ``groundspeed_kt`` and nominal ADS-B update interval
    ``cadence_s`` seconds, the aircraft travels ``groundspeed * cadence`` between
    consecutive samples, so the along-runway position can be pinned only to
    within that spacing. This is the irreducible error floor (Req 13.0): no
    estimator can drive the distance error below it purely from sample geometry.

    Both inputs are taken in magnitude so a nonsensical negative speed/cadence
    still yields a non-negative floor. A non-finite input yields ``nan``.
    """
    if not (np.isfinite(groundspeed_kt) and np.isfinite(cadence_s)):
        return float("nan")
    return abs(float(groundspeed_kt)) * KNOTS_TO_FT_PER_S * abs(float(cadence_s))


@dataclass(frozen=True)
class ErrorFloorReport:
    """The cadence-limited floor and the observed error distribution against it.

    ``floor_ft`` is the floor at the ``representative_groundspeed_kt`` (the caller
    supplied speed, else the median per-flight touchdown groundspeed).
    ``per_flight_floor_median_ft`` is the median of each flight's own floor (at
    its own groundspeed). ``fraction_within_floor`` is the fraction of flights
    whose absolute distance error is at or below their own per-flight floor --
    i.e. flights already at the irreducible resolution, where no method could
    have done better. ``observed_*`` summarize the realized error distribution.
    """

    cadence_s: float
    representative_groundspeed_kt: float
    floor_ft: float
    per_flight_floor_median_ft: float
    n_flights: int
    observed_rmse_ft: float
    observed_median_abs_error_ft: float
    observed_p95_abs_error_ft: float
    fraction_within_floor: float


def characterize_error_floor(
    evaluations: Sequence[FlightEvaluation],
    config: ValidationConfig,
    cadence_s: float,
    *,
    representative_groundspeed_kt: Optional[float] = None,
) -> ErrorFloorReport:
    """Characterize and report the cadence-limited error floor (Req 13.0, 13.1).

    Reuses :func:`~tdz.validation.metrics.compute_metrics` for the observed error
    summary. ``cadence_s`` is the nominal ADS-B update interval (pass
    ``config.uncertainty.nominal_cadence_s`` -- reused, not duplicated). When
    ``representative_groundspeed_kt`` is omitted the headline floor is taken at
    the median per-flight touchdown groundspeed. ``fraction_within_floor``
    compares each flight against its **own** per-flight floor.
    """
    included = [e for e in evaluations if e.result.confidence != NO_ESTIMATE]
    metrics = compute_metrics(included)

    speeds = np.asarray(
        [float(e.result.groundspeed_at_touchdown_kt) for e in included], dtype=float
    )
    finite_speeds = speeds[np.isfinite(speeds)]

    if representative_groundspeed_kt is not None:
        rep_speed = float(representative_groundspeed_kt)
    elif finite_speeds.size:
        rep_speed = float(np.median(finite_speeds))
    else:
        rep_speed = float("nan")

    floor_ft = cadence_limited_floor_ft(rep_speed, cadence_s)

    # Per-flight floors and the fraction of flights already at/under their floor.
    per_flight_floor: list[float] = []
    within = 0
    counted = 0
    for ev in included:
        v = float(ev.result.groundspeed_at_touchdown_kt)
        fl = cadence_limited_floor_ft(v, cadence_s)
        if not np.isfinite(fl):
            continue
        per_flight_floor.append(fl)
        abs_err = abs(
            float(ev.result.along_runway_distance_ft)
            - _truth_distance_ft(ev)
        )
        if np.isfinite(abs_err):
            counted += 1
            if abs_err <= fl:
                within += 1

    pff = np.asarray(per_flight_floor, dtype=float)
    per_flight_floor_median = float(np.median(pff)) if pff.size else float("nan")
    fraction_within = (within / counted) if counted else float("nan")

    return ErrorFloorReport(
        cadence_s=float(cadence_s),
        representative_groundspeed_kt=rep_speed,
        floor_ft=floor_ft,
        per_flight_floor_median_ft=per_flight_floor_median,
        n_flights=int(metrics.n_flights),
        observed_rmse_ft=float(metrics.distance_rmse_ft),
        observed_median_abs_error_ft=float(metrics.distance_median_abs_error_ft),
        observed_p95_abs_error_ft=float(metrics.distance_p95_abs_error_ft),
        fraction_within_floor=fraction_within,
    )


def _truth_distance_ft(ev: FlightEvaluation) -> float:
    """Clock-independent along-runway distance truth in feet for one flight."""
    from tdz.validation.metrics import along_runway_truth_distance_ft

    return along_runway_truth_distance_ft(
        ev.runway, ev.truth.touchdown_lat, ev.truth.touchdown_lon
    )


# ---------------------------------------------------------------------------
# Below-target flagging (Req 13.2-13.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BelowTargetFlag:
    """A single below-target finding for one stratum and one metric.

    ``dimension`` / ``key`` identify the stratum (e.g. ``"source"`` / ``"aireon"``);
    ``metric`` names the failing quantity; ``observed`` and ``target`` are the
    measured value and the provisional target; ``direction`` is ``"max"`` when the
    metric must stay at or below the target (RMSE, p95, bias magnitude) or
    ``"min"`` when it must stay at or above it. A flag is advisory only -- it is
    never raised as an error (targets are provisional, Req 13.0).
    """

    dimension: str
    key: str
    n_flights: int
    metric: str
    observed: float
    target: float
    direction: str


def _stratum_flags(
    dimension: str,
    key: str,
    n_flights: int,
    metrics,
    targets,
) -> list[BelowTargetFlag]:
    """Return the below-target flags for one stratum (Req 13.1, 13.2, 13.3).

    Only criteria 1-3 of Req 13 drive flagging (Req 13.5). NaN metrics -- e.g. a
    long-side percentile with no positive errors -- are treated as "no data",
    not a failure, and are skipped.
    """
    flags: list[BelowTargetFlag] = []

    def _max_check(metric_name: str, observed: float, target: float) -> None:
        if np.isfinite(observed) and observed > target:
            flags.append(
                BelowTargetFlag(
                    dimension=dimension,
                    key=key,
                    n_flights=n_flights,
                    metric=metric_name,
                    observed=float(observed),
                    target=float(target),
                    direction="max",
                )
            )

    # Req 13.1: overall along-runway RMSE.
    _max_check("distance_rmse_ft", metrics.distance_rmse_ft, targets.distance_rmse_ft)
    # Req 13.3: 95th-percentile absolute error and long-side (positive) cap.
    _max_check(
        "distance_p95_abs_error_ft",
        metrics.distance_p95_abs_error_ft,
        targets.distance_p95_abs_error_ft,
    )
    _max_check(
        "distance_p95_long_side_ft",
        metrics.distance_p95_long_side_ft,
        targets.distance_p95_long_side_ft,
    )
    # Req 13.2: |median signed error| bias cap.
    if np.isfinite(metrics.distance_median_signed_error_ft):
        bias = abs(float(metrics.distance_median_signed_error_ft))
        if bias > targets.median_signed_error_abs_ft:
            flags.append(
                BelowTargetFlag(
                    dimension=dimension,
                    key=key,
                    n_flights=n_flights,
                    metric="distance_median_signed_error_abs_ft",
                    observed=bias,
                    target=float(targets.median_signed_error_abs_ft),
                    direction="max",
                )
            )

    return flags


def flag_below_target(
    report: StratifiedMetricsReport,
    config: ValidationConfig,
    *,
    dimensions: Sequence[str] = _FLAG_DIMENSIONS,
) -> tuple[BelowTargetFlag, ...]:
    """Flag below-target strata against the provisional targets (Req 13.2-13.5).

    Inspects every stratum in ``report`` (both reportable and below-reporting-gate
    strata) on the given ``dimensions`` (ADS-B source and aircraft type per Req
    13.5) and flags those whose metrics miss the provisional targets -- but only
    when the stratum holds at least ``config.below_target_min_flights`` flights
    (default 200; distinct from the >=30 reporting gate). Strata below that gate
    are never flagged.

    This is **advisory**: the targets are provisional until ratified against the
    cadence-limited error floor (Req 13.0), so the function returns a tuple of
    flags and never raises. Flags are ordered deterministically by
    (dimension, key, metric).
    """
    min_flights = int(config.below_target_min_flights)
    targets = config.provisional_targets
    dims = set(dimensions)

    flags: list[BelowTargetFlag] = []
    for stratum in (*report.strata, *report.below_threshold):
        if stratum.dimension not in dims:
            continue
        if stratum.n_flights < min_flights:
            continue
        flags.extend(
            _stratum_flags(
                stratum.dimension,
                stratum.key,
                stratum.n_flights,
                stratum.metrics,
                targets,
            )
        )

    flags.sort(key=lambda f: (f.dimension, f.key, f.metric))
    return tuple(flags)
