"""Clock-independent distance truth + stratified validation metrics (Task 22.2).

This module turns paired system outputs and QAR ground truth into the error
metrics the validation harness reports (Req 12.1, 12.5-12.7, 12.9, 12.10) and
places them side-by-side with the naive first-on-ground baseline (Req 12.8 /
13.4).

Distance truth is clock-independent (Req 12.10)
------------------------------------------------
The along-runway distance TRUTH is a purely geometric quantity: the QAR
touchdown lat/long projected onto the runway centerline via the shared
:class:`~tdz.geo.RunwayProjector` primitive. It therefore never depends on the
QAR<->ADS-B clock alignment (Task 21) -- no clock offset is applied to the
distance truth. Clock alignment matters only for the time-error metric, whose
truth is the (already clock-aligned) QAR touchdown time; flights whose clock
offset could not be reliably estimated (``clock_offset_quality == "failed"``)
are dropped from the time metric yet retained for distance metrics (Req 19.6/7,
12.10).

Sign convention (Req 12.5)
--------------------------
Signed distance error is ``system - truth`` in feet: **positive means the
estimate is longer** (farther past the threshold) than truth. The long-side
(positive) tail is the overrun-hazard tail the system exists to surface, so it
is reported separately as the 95th-percentile positive signed error (Req 12.6).

Units
-----
Internal geometry is SI (meters); distances are reported in feet using the
shared :data:`~tdz.uncertainty.M_TO_FT` constant, matching the single SI->feet
conversion boundary used elsewhere (e.g. :mod:`tdz.assemble`). Times are in
seconds. Speeds (for the approach-speed band) are in knots, matching
:attr:`~tdz.models.TouchdownResult.groundspeed_at_touchdown_kt`.

Stratification (Req 12.7, 12.9)
-------------------------------
Metrics are computed overall and stratified by aircraft type, ADS-B source,
airport, and approach-speed band. A stratum is only *reportable* when it holds
at least ``config.validation.min_stratum_size`` flights (default 30); smaller
strata are still computed but returned under ``below_threshold`` and marked
``reportable=False`` so they are never presented as reliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from tdz.config.schema import ValidationConfig
from tdz.geo import project_to_runway
from tdz.models import QARTruthRecord, RunwayReference, TouchdownResult, ValidationMetrics
from tdz.uncertainty import M_TO_FT
from tdz.validation.clock_alignment import QUALITY_FAILED

__all__ = [
    "NO_ESTIMATE",
    "FlightEvaluation",
    "StratumResult",
    "StratifiedMetricsReport",
    "along_runway_truth_distance_ft",
    "approach_speed_band_label",
    "compute_metrics",
    "compute_stratified_metrics",
]

#: Confidence classification excluded from the metrics (no touchdown produced).
NO_ESTIMATE: str = "no-estimate"


# ---------------------------------------------------------------------------
# Per-flight evaluation input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlightEvaluation:
    """One flight's system output paired with its QAR truth and runway geometry.

    ``result`` is the system :class:`~tdz.models.TouchdownResult`; ``truth`` is
    the (optionally clock-aligned) :class:`~tdz.models.QARTruthRecord`; ``runway``
    is the :class:`~tdz.models.RunwayReference` used to project the truth
    lat/long onto the centerline for the clock-independent distance truth.

    ``baseline_distance_m`` is the naive first-on-ground along-runway distance
    (meters from threshold) produced by
    :func:`tdz.pipeline.naive_baseline_distance`; pass ``None`` when the baseline
    was unavailable for this flight (no on-ground sample), in which case the
    flight simply does not contribute to the baseline comparison.
    """

    result: TouchdownResult
    truth: QARTruthRecord
    runway: RunwayReference
    baseline_distance_m: Optional[float] = None


# ---------------------------------------------------------------------------
# Stratified report containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StratumResult:
    """Metrics for a single stratum, with the reportability gate applied.

    ``dimension`` is the stratification axis (``"aircraft_type"``, ``"source"``,
    ``"airport"``, ``"approach_speed_band"``); ``key`` is the stratum value.
    ``reportable`` is ``True`` iff ``n_flights >= min_stratum_size`` (Req 12.9).
    """

    dimension: str
    key: str
    n_flights: int
    reportable: bool
    metrics: ValidationMetrics


@dataclass(frozen=True)
class StratifiedMetricsReport:
    """Overall + stratified metrics with the system-vs-baseline comparison.

    ``overall`` carries the corpus metrics (system stats plus ``baseline_rmse_ft``
    and ``improvement_pct`` for the side-by-side comparison, Req 12.8). ``strata``
    holds the reportable strata (>= ``min_stratum_size``); ``below_threshold``
    holds the suppressed ones. ``min_stratum_size`` and
    ``approach_speed_band_edges_kt`` record the gate and banding used.
    """

    overall: ValidationMetrics
    strata: tuple[StratumResult, ...]
    below_threshold: tuple[StratumResult, ...]
    min_stratum_size: int
    approach_speed_band_edges_kt: tuple[float, ...]


# ---------------------------------------------------------------------------
# Distance truth + banding helpers
# ---------------------------------------------------------------------------


def along_runway_truth_distance_ft(
    runway: RunwayReference, truth_lat: float, truth_lon: float
) -> float:
    """Clock-independent along-runway distance truth in feet (Req 12.10).

    Projects the QAR touchdown lat/long onto the runway centerline with the
    shared :class:`~tdz.geo.RunwayProjector` and converts the SI result to feet.
    Because this is a pure geodesic projection, it never depends on the
    QAR<->ADS-B clock alignment.
    """
    projected = project_to_runway(runway, float(truth_lat), float(truth_lon))
    return projected.along_runway_distance_m * M_TO_FT


def approach_speed_band_label(speed_kt: float, edges: Sequence[float]) -> str:
    """Label the approach-speed band containing ``speed_kt`` (knots).

    ``edges`` is a strictly-increasing sequence of band boundaries; ``N`` edges
    define ``N+1`` bands labelled ``"<e0"``, ``"e0-e1"``, ..., ``">=e_{N-1}"``.
    A non-finite speed maps to ``"unknown"``. Integer-valued edges are rendered
    without a trailing ``.0`` for compact, stable stratum keys.
    """
    if not np.isfinite(speed_kt):
        return "unknown"

    def _fmt(x: float) -> str:
        return str(int(x)) if float(x).is_integer() else f"{x:g}"

    ordered = list(edges)
    if not ordered:
        return "all"
    if speed_kt < ordered[0]:
        return f"<{_fmt(ordered[0])}"
    for lo, hi in zip(ordered, ordered[1:]):
        if lo <= speed_kt < hi:
            return f"{_fmt(lo)}-{_fmt(hi)}"
    return f">={_fmt(ordered[-1])}"


# ---------------------------------------------------------------------------
# Scalar metric helpers
# ---------------------------------------------------------------------------


def _rmse(errors: np.ndarray) -> float:
    """Root-mean-square error, or ``nan`` for an empty input."""
    if errors.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(errors * errors)))


def _pct(values: np.ndarray, q: float) -> float:
    """``np.percentile`` guarding the empty case (returns ``nan``)."""
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def _median(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(values))


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    evaluations: Sequence[FlightEvaluation],
    *,
    stratum_key: Optional[str] = None,
) -> ValidationMetrics:
    """Compute :class:`~tdz.models.ValidationMetrics` over a set of flights.

    Flights whose confidence classification is ``"no-estimate"`` produced no
    touchdown and are excluded. Distance metrics use every remaining flight;
    the time-error metric additionally excludes flights whose QAR clock offset
    could not be reliably estimated (``clock_offset_quality == "failed"``),
    since their aligned time is untrustworthy (Req 12.10, 19.6). The baseline
    comparison uses only flights that carry a ``baseline_distance_m``.

    All distance quantities are feet, times are seconds. Signed distance error
    is ``system - truth`` (positive = longer; Req 12.5).
    """
    included = [e for e in evaluations if e.result.confidence != NO_ESTIMATE]

    signed_err: list[float] = []
    truth_ft_list: list[float] = []
    ci_lo_list: list[float] = []
    ci_hi_list: list[float] = []
    time_err: list[float] = []
    baseline_err: list[float] = []

    for ev in included:
        truth_ft = along_runway_truth_distance_ft(
            ev.runway, ev.truth.touchdown_lat, ev.truth.touchdown_lon
        )
        system_ft = float(ev.result.along_runway_distance_ft)
        signed_err.append(system_ft - truth_ft)
        truth_ft_list.append(truth_ft)
        ci_lo_list.append(float(ev.result.distance_ci_90_lower_ft))
        ci_hi_list.append(float(ev.result.distance_ci_90_upper_ft))

        # Time error against the clock-aligned QAR touchdown time; skip flights
        # with an unreliable clock offset (retained for distance only).
        if ev.truth.clock_offset_quality != QUALITY_FAILED:
            t_err = float(ev.result.touchdown_time) - float(ev.truth.touchdown_time_qar)
            if np.isfinite(t_err):
                time_err.append(t_err)

        if ev.baseline_distance_m is not None:
            baseline_ft = float(ev.baseline_distance_m) * M_TO_FT
            baseline_err.append(baseline_ft - truth_ft)

    signed = np.asarray(signed_err, dtype=float)
    abs_err = np.abs(signed)
    truth_ft_arr = np.asarray(truth_ft_list, dtype=float)
    ci_lo = np.asarray(ci_lo_list, dtype=float)
    ci_hi = np.asarray(ci_hi_list, dtype=float)
    t_err_arr = np.asarray(time_err, dtype=float)
    base = np.asarray(baseline_err, dtype=float)

    # Long-side (positive) signed errors -> 95th percentile of the overrun tail.
    long_side = signed[signed > 0.0]
    p95_long_side = _pct(long_side, 95.0)

    # 90% distance-CI coverage: truth inside [lower, upper] (clock-independent).
    if truth_ft_arr.size:
        inside = (truth_ft_arr >= ci_lo) & (truth_ft_arr <= ci_hi)
        coverage = float(np.count_nonzero(inside) / truth_ft_arr.size)
    else:
        coverage = float("nan")

    system_rmse = _rmse(signed)
    baseline_rmse = _rmse(base)
    if base.size and np.isfinite(baseline_rmse) and baseline_rmse > 0.0:
        improvement = (baseline_rmse - system_rmse) / baseline_rmse * 100.0
    else:
        improvement = float("nan")

    return ValidationMetrics(
        n_flights=int(signed.size),
        distance_rmse_ft=system_rmse,
        distance_median_abs_error_ft=_median(abs_err),
        distance_iqr_ft=(_pct(abs_err, 25.0), _pct(abs_err, 75.0)),
        distance_p95_abs_error_ft=_pct(abs_err, 95.0),
        distance_p99_abs_error_ft=_pct(abs_err, 99.0),
        distance_p95_long_side_ft=p95_long_side,
        distance_median_signed_error_ft=_median(signed),
        time_rmse_s=_rmse(t_err_arr),
        time_median_abs_error_s=_median(np.abs(t_err_arr)),
        baseline_rmse_ft=baseline_rmse,
        improvement_pct=improvement,
        ci_90_coverage=coverage,
        stratum_key=stratum_key,
    )


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------


def _stratum_value(ev: FlightEvaluation, dimension: str, edges: Sequence[float]) -> str:
    """Return the stratum key for ``ev`` on the given ``dimension``."""
    if dimension == "aircraft_type":
        return str(ev.result.aircraft_type)
    if dimension == "source":
        return str(ev.result.ads_b_source)
    if dimension == "airport":
        return str(ev.truth.airport_id)
    if dimension == "approach_speed_band":
        return approach_speed_band_label(
            float(ev.result.groundspeed_at_touchdown_kt), edges
        )
    raise ValueError(f"unknown stratification dimension '{dimension}'")


#: The stratification axes reported by the harness (Req 12.7).
_DIMENSIONS: tuple[str, ...] = (
    "aircraft_type",
    "source",
    "airport",
    "approach_speed_band",
)


def compute_stratified_metrics(
    evaluations: Sequence[FlightEvaluation],
    config: ValidationConfig,
) -> StratifiedMetricsReport:
    """Compute overall + stratified metrics with the baseline comparison.

    Produces the corpus (overall) metrics and, for each stratification axis
    (aircraft type, ADS-B source, airport, approach-speed band; Req 12.7),
    per-stratum metrics gated by ``config.min_stratum_size`` (Req 12.9). Each
    :class:`~tdz.models.ValidationMetrics` already carries both the system stats
    and the naive-baseline RMSE + improvement, giving the required side-by-side
    system-vs-baseline reporting (Req 12.8).

    Strata are visited in a deterministic order: dimension order followed by
    sorted stratum key. Reportable strata (>= ``min_stratum_size``) are returned
    in ``strata``; smaller strata are returned in ``below_threshold`` and flagged
    ``reportable=False`` rather than being presented as reliable.
    """
    included = [e for e in evaluations if e.result.confidence != NO_ESTIMATE]
    edges = tuple(config.approach_speed_band_edges_kt)
    min_size = int(config.min_stratum_size)

    overall = compute_metrics(included)

    reportable: list[StratumResult] = []
    suppressed: list[StratumResult] = []

    for dimension in _DIMENSIONS:
        groups: dict[str, list[FlightEvaluation]] = {}
        for ev in included:
            groups.setdefault(_stratum_value(ev, dimension, edges), []).append(ev)

        for key in sorted(groups):
            members = groups[key]
            metrics = compute_metrics(members, stratum_key=key)
            n = metrics.n_flights
            is_reportable = n >= min_size
            entry = StratumResult(
                dimension=dimension,
                key=key,
                n_flights=n,
                reportable=is_reportable,
                metrics=metrics,
            )
            (reportable if is_reportable else suppressed).append(entry)

    return StratifiedMetricsReport(
        overall=overall,
        strata=tuple(reportable),
        below_threshold=tuple(suppressed),
        min_stratum_size=min_size,
        approach_speed_band_edges_kt=edges,
    )
