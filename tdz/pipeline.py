"""Stage-1-3 baseline wiring + naive baseline + distance-error harness (Task 14).

This module is **integration / wiring**, not a new estimator. It connects the
pieces already built in Tasks 1-13 into a runnable stage-1-3 pipeline over a
single flight and produces a *preliminary* along-runway distance-error
distribution so the cadence-limited error floor can be surfaced (Req 13.0) and
the system can be shown to beat the naive "first on-ground sample position"
strawman (Req 12.8).

Pipeline wired here (per flight)
--------------------------------
1. **Classify + coarse-bracket** the trajectory
   (:func:`tdz.bracket.compute_coarse_bracket`). A go-around (or suppressed
   touch-and-go) short-circuits to a no-touchdown :class:`StageRunResult` with
   no combined distance -- the runner never fabricates an estimate for a
   trajectory that did not land.
2. **QA-gate** the record against the coarse bracket window
   (:func:`tdz.io.qa.run_qa`); estimators consume the cleaned record.
3. **Source-gate** the estimator set (:func:`tdz.io.gating.gate_estimators`) so
   geometric-altitude estimators (``flare_crossing`` / ``imm_rts``) are excluded
   for a source without true HAE (e.g. the FR24 assumption), then run each
   eligible physics + change-point estimator's ``.estimate(flight)``.
4. **Provisionally combine** the non-failed eligible estimates into a single
   ``t_td`` (see :func:`combine_estimates`).
5. **Map** the combined ``t_td`` to an along-runway touchdown *distance* by
   interpolating the trajectory position at ``t_td``
   (:func:`tdz.timebase.interpolation.interpolate_position_at`) and projecting
   onto the runway centerline (:func:`tdz.geo.project_to_runway`), optionally
   applying the lever-arm along-runway correction.

PROVISIONAL combiner -- explicitly NOT the Task-18 fusion
---------------------------------------------------------
:func:`combine_estimates` is a deliberately simple **inverse-variance-weighted
mean** of the non-failed, source-eligible estimator ``t_td`` values
(``weight_i = 1 / sigma_i^2``). It exists only to produce a single number per
flight so a distance-error distribution can be measured for this first
milestone. It is **not** the calibrated fusion ensemble (Task 18): there is no
stacking, no reliability weighting, no calibrated predictive interval, no
disagreement/CI gating. Those arrive in Tasks 18-20. The combiner does honour
the one hard invariant that is already enforced upstream -- every estimator's
``t_td`` is bounded above by the on-ground transition (Property 5 /
:class:`tdz.estimators.physics.PhysicsEstimator`) -- and gives the on-ground
flag itself zero weight (it is only ever the bracket's upper bound).

Distance-error harness -- PRELIMINARY synthetic characterisation
----------------------------------------------------------------
:func:`distance_error_summary` compares the provisional system distance against
the naive baseline distance over a set of flights with **known** synthetic truth
touchdown positions, computing the along-runway truth distance directly from the
truth lat/long (a clock-independent geometric quantity, Req 12.10) and reporting
RMSE / median / p95 absolute error plus the system-vs-baseline RMSE improvement
(Req 12.8 / 13.4). This is a *preliminary* synthetic read on the cadence-limited
floor only; the full QAR validation harness (grouped splits, stratified metrics,
coverage, cross-source) is Task 22, and the SI->feet output boundary is Task 20.

Units convention
----------------
SI throughout: times in epoch seconds, distances in metres. No conversion to
feet/knots happens here (that is the output boundary, Task 20). All reported
distance metrics are therefore in metres; callers that report feet convert at
their boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping, Optional, Sequence

import numpy as np

from tdz.bracket import BracketResult, compute_coarse_bracket
from tdz.config.models import SourceCapability
from tdz.config.schema import TDZConfig
from tdz.estimators.changepoint import (
    CusumEstimator,
    GlrtEstimator,
    JerkOnsetEstimator,
    PeltEstimator,
)
from tdz.estimators.physics import (
    DecelKneeEstimator,
    FlareCrossingEstimator,
    ImmRtsEstimator,
)
from tdz.geo import LeverArmCorrection, project_to_runway
from tdz.io.gating import SourceGating, gate_estimators
from tdz.io.qa import QAResult, run_qa
from tdz.models import BaseEstimator, FailureReason, FlightRecord, TDEstimate
from tdz.timebase.interpolation import interpolate_position_at

__all__ = [
    "PHYSICS_ESTIMATOR_IDS",
    "CHANGEPOINT_ESTIMATOR_IDS",
    "ESTIMATOR_REGISTRY",
    "CombinedEstimate",
    "StageRunResult",
    "NaiveBaselineResult",
    "FlightTruth",
    "BaselineComparison",
    "build_estimators",
    "combine_estimates",
    "run_stage123",
    "naive_baseline_distance",
    "along_runway_truth_distance",
    "distance_error_summary",
]


# ---------------------------------------------------------------------------
# Estimator registry (physics + change-point families only)
# ---------------------------------------------------------------------------

#: Physics estimator ids (Task 12), in a stable order.
PHYSICS_ESTIMATOR_IDS: Final[tuple[str, ...]] = (
    "decel_knee",
    "flare_crossing",
    "imm_rts",
)

#: Change-point estimator ids (Task 13), in a stable order.
CHANGEPOINT_ESTIMATOR_IDS: Final[tuple[str, ...]] = (
    "pelt",
    "cusum",
    "glrt",
    "jerk_onset",
)

#: Maps an estimator id to a zero-argument factory building a fresh instance.
#: Restricted to the physics + change-point families this baseline wires in --
#: learned estimators (Tasks 15-17) and fusion (Task 18) are intentionally
#: absent. Unknown / not-yet-built ids in ``config.estimators.enabled`` (e.g.
#: ``lightgbm``) are simply skipped by :func:`build_estimators`.
ESTIMATOR_REGISTRY: Final[dict[str, type[BaseEstimator]]] = {
    "decel_knee": DecelKneeEstimator,
    "flare_crossing": FlareCrossingEstimator,
    "imm_rts": ImmRtsEstimator,
    "pelt": PeltEstimator,
    "cusum": CusumEstimator,
    "glrt": GlrtEstimator,
    "jerk_onset": JerkOnsetEstimator,
}

#: Maps a :class:`~tdz.models.FlightRecord` ADS-B source label to the config
#: ``sources`` key (the schema uses ``"fr24"`` while records may carry the
#: long-form ``"flightradar24"``).
_SOURCE_ALIASES: Final[dict[str, str]] = {
    "aireon": "aireon",
    "fr24": "fr24",
    "flightradar24": "fr24",
}


# ---------------------------------------------------------------------------
# Result value objects (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombinedEstimate:
    """Provisional pre-fusion combined touchdown time for one flight.

    NOT the Task-18 fusion (see module docstring). ``t_td`` is the
    inverse-variance-weighted mean of the contributing estimator ``t_td`` values
    and ``sigma_t`` is the corresponding combined 1-sigma. ``ok`` is ``False``
    (with ``t_td``/``sigma_t`` ``NaN`` and a ``reason_code``) when no estimator
    contributed.
    """

    ok: bool
    t_td: float
    sigma_t: float
    contributing: tuple[str, ...]
    weights: dict[str, float]
    reason_code: Optional[FailureReason]


@dataclass(frozen=True)
class StageRunResult:
    """Outcome of the stage-1-3 baseline run for a single flight.

    Attributes
    ----------
    flight_id, ads_b_source:
        Echoed from the input record.
    runway:
        The runway reference used for projection (carried so the distance-error
        harness can compute the clock-independent truth distance).
    bracket:
        The coarse-bracket result (carries the trajectory classification and the
        no-touchdown disposition).
    qa_status, qa_reason:
        The QA gate status (``"ok"`` / ``"no-estimate"``) and reason code.
    gating:
        The source-capability gating decision (eligible / excluded estimators).
    estimates:
        ``{estimator_id: TDEstimate}`` for every eligible estimator that was run
        (failed estimates included, for traceability).
    combined:
        The provisional combined estimate (see :class:`CombinedEstimate`).
    combined_t_td, combined_distance_m, lateral_offset_m:
        Convenience accessors: the combined ``t_td`` (epoch s), its along-runway
        touchdown distance (m, lever-arm-corrected when supplied) and the lateral
        offset (m). ``None`` when no combined estimate was produced or the
        trajectory did not land.
    naive_baseline:
        The naive first-on-ground baseline for this flight (also available
        standalone via :func:`naive_baseline_distance`).
    no_touchdown:
        ``True`` when the trajectory short-circuited to a no-touchdown result
        (go-around / suppressed touch-and-go); no combined distance is produced.
    reason_code:
        The governing failure reason when no usable estimate was produced.
    diagnostics:
        Free-form diagnostics (interpolation degraded flag, lever-arm applied).
    """

    flight_id: str
    ads_b_source: str
    runway: object
    bracket: BracketResult
    qa_status: str
    qa_reason: Optional[FailureReason]
    gating: SourceGating
    estimates: dict[str, TDEstimate]
    combined: CombinedEstimate
    combined_t_td: Optional[float]
    combined_distance_m: Optional[float]
    lateral_offset_m: Optional[float]
    naive_baseline: "NaiveBaselineResult"
    no_touchdown: bool
    reason_code: Optional[FailureReason]
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NaiveBaselineResult:
    """The naive "first on-ground sample position" baseline (Req 12.8).

    The strawman the system must beat: the along-runway distance of the FIRST
    position sample whose on-ground flag is ``True``. Because the on-ground flag
    transitions *after* real touchdown, this distance systematically overshoots
    down the runway, which is exactly why it is a baseline rather than an
    estimate.

    Attributes
    ----------
    available:
        ``True`` when at least one on-ground sample existed.
    distance_m:
        Along-runway distance (m from threshold) of the first on-ground sample,
        or ``None`` when unavailable.
    lateral_offset_m:
        Lateral offset (m) of that sample, or ``None``.
    sample_time:
        The position time (epoch s) of the first on-ground sample, or ``None``.
    sample_index:
        Index into the (cleaned) position arrays, or ``None``.
    """

    available: bool
    distance_m: Optional[float]
    lateral_offset_m: Optional[float]
    sample_time: Optional[float]
    sample_index: Optional[int]


@dataclass(frozen=True)
class FlightTruth:
    """Known synthetic truth touchdown position for one flight (Req 12.10).

    The truth along-runway distance is computed directly from
    ``touchdown_lat`` / ``touchdown_lon`` (a clock-independent geometric
    quantity), so the distance metric never depends on clock alignment.
    """

    flight_id: str
    touchdown_lat: float
    touchdown_lon: float


@dataclass(frozen=True)
class BaselineComparison:
    """Preliminary along-runway distance-error distribution (Req 12.8 / 13.0).

    All distances are in METRES (the SI->feet conversion is the Task-20 output
    boundary). Computed only over flights where BOTH the provisional system
    estimate and the naive baseline produced a distance and a truth was supplied
    (``n_flights``), so the comparison is like-for-like.

    This is a PRELIMINARY synthetic characterisation of the cadence-limited error
    floor, not a ratified accuracy result: the full QAR validation harness
    (grouped splits, stratified metrics, coverage) is Task 22.

    Attributes
    ----------
    n_flights:
        Number of flights contributing to the comparison.
    system_rmse_m, system_median_abs_error_m, system_p95_abs_error_m,
    system_median_signed_error_m:
        System (provisional combined) error distribution. Signed error is
        ``estimate - truth`` (positive = long, past the threshold).
    baseline_rmse_m, baseline_median_abs_error_m, baseline_p95_abs_error_m,
    baseline_median_signed_error_m:
        Naive-baseline error distribution, same conventions.
    rmse_improvement_pct:
        ``(baseline_rmse - system_rmse) / baseline_rmse * 100``; positive means
        the system beats the strawman (Req 13.4 target: >= 30%).
    cadence_limited_floor_m:
        A preliminary read on the irreducible cadence-limited floor: the system
        RMSE on this clean synthetic set (with no sensor noise the residual error
        is dominated by the 4-5 s sampling interval). Reported, not ratified
        (Req 13.0).
    system_errors_m, baseline_errors_m:
        The per-flight signed errors (m), aligned, for inspection / plotting.
    """

    n_flights: int
    system_rmse_m: float
    system_median_abs_error_m: float
    system_p95_abs_error_m: float
    system_median_signed_error_m: float
    baseline_rmse_m: float
    baseline_median_abs_error_m: float
    baseline_p95_abs_error_m: float
    baseline_median_signed_error_m: float
    rmse_improvement_pct: float
    cadence_limited_floor_m: float
    system_errors_m: tuple[float, ...]
    baseline_errors_m: tuple[float, ...]


# ---------------------------------------------------------------------------
# Estimator construction & provisional combiner
# ---------------------------------------------------------------------------


def build_estimators(config: TDZConfig) -> dict[str, BaseEstimator]:
    """Instantiate the enabled physics + change-point estimators.

    Returns ``{estimator_id: instance}`` for every id in
    ``config.estimators.enabled`` that is present in :data:`ESTIMATOR_REGISTRY`,
    preserving the configured order. Ids the registry does not know about
    (learned estimators / fusion, not wired by this baseline) are skipped.
    """
    estimators: dict[str, BaseEstimator] = {}
    for name in config.estimators.enabled:
        factory = ESTIMATOR_REGISTRY.get(name)
        if factory is not None:
            estimators[name] = factory()
    return estimators


def combine_estimates(
    estimates: Mapping[str, TDEstimate],
    eligible_ids: Sequence[str],
) -> CombinedEstimate:
    """Provisionally combine eligible, non-failed estimates (NOT Task-18 fusion).

    Computes the inverse-variance-weighted mean of the ``t_td`` values of the
    eligible estimators that did not fail and reported a finite, positive
    ``sigma_t``::

        weight_i = 1 / sigma_i^2
        t_td     = sum(weight_i * t_i) / sum(weight_i)
        sigma_t  = sqrt(1 / sum(weight_i))

    This is a placeholder combiner to produce one number per flight for the
    distance-error distribution; it is explicitly **not** the calibrated fusion
    ensemble (no stacking, reliability weighting, calibrated interval, or
    disagreement/CI gating -- those are Task 18). The on-ground flag is given no
    weight here (it only ever bounds the bracket).

    Parameters
    ----------
    estimates:
        ``{estimator_id: TDEstimate}`` produced for the flight.
    eligible_ids:
        Source-eligible estimator ids (from :class:`SourceGating`); ids outside
        this set are ignored even if present in ``estimates``.

    Returns
    -------
    CombinedEstimate
        ``ok=True`` with the combined ``t_td`` / ``sigma_t`` and contributing
        ids, or ``ok=False`` with :attr:`FailureReason.ALL_ESTIMATORS_FAILED`
        when nothing contributed.
    """
    eligible = set(eligible_ids)
    contributing: list[str] = []
    weights: dict[str, float] = {}
    times: list[float] = []

    for name in eligible_ids:
        est = estimates.get(name)
        if est is None or name not in eligible:
            continue
        if est.confidence == "failed":
            continue
        sigma = float(est.sigma_t)
        t = float(est.t_td)
        if not (np.isfinite(sigma) and sigma > 0.0 and np.isfinite(t)):
            continue
        w = 1.0 / (sigma * sigma)
        contributing.append(name)
        weights[name] = w
        times.append(t)

    if not contributing:
        return CombinedEstimate(
            ok=False,
            t_td=float("nan"),
            sigma_t=float("nan"),
            contributing=(),
            weights={},
            reason_code=FailureReason.ALL_ESTIMATORS_FAILED,
        )

    w_arr = np.array([weights[n] for n in contributing], dtype=float)
    t_arr = np.array(times, dtype=float)
    w_sum = float(np.sum(w_arr))
    t_td = float(np.sum(w_arr * t_arr) / w_sum)
    sigma_t = float(np.sqrt(1.0 / w_sum))

    return CombinedEstimate(
        ok=True,
        t_td=t_td,
        sigma_t=sigma_t,
        contributing=tuple(contributing),
        weights=weights,
        reason_code=None,
    )


# ---------------------------------------------------------------------------
# Source-capability resolution
# ---------------------------------------------------------------------------


def _resolve_source_capability(
    flight: FlightRecord,
    config: TDZConfig,
    override: Optional[SourceCapability],
) -> SourceCapability:
    """Resolve the :class:`SourceCapability` for a flight's ADS-B source.

    Uses ``override`` when supplied, else looks the flight's source up in
    ``config.sources`` (mapping the long-form ``flightradar24`` label to the
    ``fr24`` config key). Falls back to a permissive aireon-like capability when
    the source is unknown, so the runner stays usable in tests/fixtures that do
    not enumerate every source.
    """
    if override is not None:
        return override
    key = _SOURCE_ALIASES.get(flight.ads_b_source, flight.ads_b_source)
    capability = config.sources.get(key)
    if capability is not None:
        return capability
    return SourceCapability(
        source=flight.ads_b_source,
        has_geometric_altitude=True,
        samples_are_raw=True,
        async_timestamps=True,
    )


# ---------------------------------------------------------------------------
# Position mapping helpers
# ---------------------------------------------------------------------------


def _map_time_to_distance(
    flight: FlightRecord,
    t_td: float,
    *,
    lever_arm: Optional[LeverArmCorrection],
) -> tuple[float, float, bool]:
    """Map a touchdown time to (along_runway_distance_m, lateral_offset_m, degraded).

    Interpolates the trajectory position at ``t_td`` (kinematic dead-reckoning),
    projects it onto the runway centerline, and -- when a lever-arm correction is
    supplied -- subtracts the along-runway shift so the distance corresponds to
    main-gear contact rather than the antenna (Task 6 sign convention). The
    lever-arm correction is OPTIONAL here; the full pitch-resolved correction and
    CI handling live at the Task-20 output boundary.
    """
    query = interpolate_position_at(
        flight.position_times,
        flight.latitudes,
        flight.longitudes,
        flight.velocity_times,
        flight.groundspeeds,
        flight.tracks,
        float(t_td),
    )
    projected = project_to_runway(flight.runway, query.lat, query.lon)
    along = projected.along_runway_distance_m
    if lever_arm is not None:
        along -= lever_arm.along_runway_shift_m
    return along, projected.lateral_offset_m, query.degraded


# ---------------------------------------------------------------------------
# Naive baseline (Req 12.8)
# ---------------------------------------------------------------------------


def naive_baseline_distance(flight: FlightRecord) -> NaiveBaselineResult:
    """Compute the naive first-on-ground-sample along-runway distance (Req 12.8).

    Finds the FIRST position sample whose on-ground flag is ``True`` and projects
    its lat/lon onto the runway centerline. This is the strawman baseline the
    system must materially outperform; it overshoots because the on-ground flag
    lags real touchdown.

    Returns
    -------
    NaiveBaselineResult
        ``available=False`` (all fields ``None``) when no on-ground sample exists
        or its coordinates are missing.
    """
    flags = np.asarray(flight.on_ground_flags, dtype=bool)
    if flags.size == 0 or not bool(np.any(flags)):
        return NaiveBaselineResult(
            available=False,
            distance_m=None,
            lateral_offset_m=None,
            sample_time=None,
            sample_index=None,
        )

    idx = int(np.argmax(flags))  # first True
    lat = float(flight.latitudes[idx])
    lon = float(flight.longitudes[idx])
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return NaiveBaselineResult(
            available=False,
            distance_m=None,
            lateral_offset_m=None,
            sample_time=float(flight.position_times[idx]),
            sample_index=idx,
        )

    projected = project_to_runway(flight.runway, lat, lon)
    return NaiveBaselineResult(
        available=True,
        distance_m=projected.along_runway_distance_m,
        lateral_offset_m=projected.lateral_offset_m,
        sample_time=float(flight.position_times[idx]),
        sample_index=idx,
    )


# ---------------------------------------------------------------------------
# Stage-1-3 runner
# ---------------------------------------------------------------------------


def run_stage123(
    flight: FlightRecord,
    config: TDZConfig,
    *,
    lever_arm: Optional[LeverArmCorrection] = None,
    source_capability: Optional[SourceCapability] = None,
) -> StageRunResult:
    """Run stages 1-3 end-to-end for one flight and provisionally combine (Task 14).

    See the module docstring for the full wiring. In short: classify + bracket ->
    QA-gate -> source-gate + run eligible physics/change-point estimators ->
    provisional inverse-variance combine -> map the combined ``t_td`` to an
    along-runway distance. A go-around (or suppressed touch-and-go) returns a
    no-touchdown result without fabricating an estimate.

    Parameters
    ----------
    flight:
        The parsed flight record.
    config:
        The resolved :class:`~tdz.config.schema.TDZConfig`. ``quality_gates``,
        ``geodesy``, ``estimators.enabled`` and ``sources`` are consumed here.
    lever_arm:
        Optional resolved :class:`~tdz.geo.LeverArmCorrection`. When supplied,
        its along-runway shift is subtracted from the antenna-projected distance
        (Task 6 convention). Optional by design -- the full pitch-resolved
        correction / CI widening is the Task-20 boundary.
    source_capability:
        Optional explicit capability override; otherwise resolved from
        ``config.sources`` by the flight's ADS-B source.

    Returns
    -------
    StageRunResult
        The bracket, QA status, per-estimator results, provisional combined
        estimate, the mapped along-runway distance, and the naive baseline.
    """
    capability = _resolve_source_capability(flight, config, source_capability)
    naive = naive_baseline_distance(flight)

    # --- Stage 1b: classify + coarse bracket ------------------------------
    bracket = compute_coarse_bracket(
        flight,
        geodesy_config=config.geodesy,
        half_width_s=config.quality_gates.window_half_width_s,
    )

    # Source-gating decision is computed regardless (records the excluded set).
    enabled_known = [n for n in config.estimators.enabled if n in ESTIMATOR_REGISTRY]
    gating = gate_estimators(capability, enabled_known)

    if bracket.status != "ok" or bracket.window is None:
        # Go-around / suppressed touch-and-go / no usable anchor: do not run
        # estimators or fabricate a distance (Req 21.2).
        empty_combined = CombinedEstimate(
            ok=False,
            t_td=float("nan"),
            sigma_t=float("nan"),
            contributing=(),
            weights={},
            reason_code=bracket.reason_code,
        )
        return StageRunResult(
            flight_id=flight.flight_id,
            ads_b_source=flight.ads_b_source,
            runway=flight.runway,
            bracket=bracket,
            qa_status="no-estimate",
            qa_reason=bracket.reason_code,
            gating=gating,
            estimates={},
            combined=empty_combined,
            combined_t_td=None,
            combined_distance_m=None,
            lateral_offset_m=None,
            naive_baseline=naive,
            no_touchdown=True,
            reason_code=bracket.reason_code,
            diagnostics={"trajectory_type": bracket.trajectory_type},
        )

    # --- Stage 1c: QA gate against the coarse bracket window ---------------
    qa: QAResult = run_qa(flight, config.quality_gates, touchdown_window=bracket.window)
    cleaned = qa.cleaned

    # --- Stage 2/3: run eligible estimators on the cleaned record ----------
    estimators = build_estimators(config)
    estimates: dict[str, TDEstimate] = {}
    for name in gating.eligible_estimators:
        estimator = estimators.get(name)
        if estimator is None:
            continue
        estimates[name] = estimator.estimate(cleaned)

    combined = combine_estimates(estimates, gating.eligible_estimators)

    diagnostics: dict = {
        "trajectory_type": bracket.trajectory_type,
        "excluded_estimators": list(gating.excluded_estimators),
        "lever_arm_applied": lever_arm is not None,
    }

    # If QA rejected the flight, surface that as the reason but still expose any
    # combined estimate computed (diagnostic-friendly; the gate decision is the
    # authority on usability).
    reason_code: Optional[FailureReason] = None
    if qa.status != "ok":
        reason_code = qa.reason_code

    combined_t_td: Optional[float] = None
    combined_distance_m: Optional[float] = None
    lateral_offset_m: Optional[float] = None
    if combined.ok:
        combined_t_td = combined.t_td
        dist, lateral, degraded = _map_time_to_distance(
            cleaned, combined.t_td, lever_arm=lever_arm
        )
        combined_distance_m = dist
        lateral_offset_m = lateral
        diagnostics["interpolation_degraded"] = degraded
    elif reason_code is None:
        reason_code = combined.reason_code

    return StageRunResult(
        flight_id=flight.flight_id,
        ads_b_source=flight.ads_b_source,
        runway=flight.runway,
        bracket=bracket,
        qa_status=qa.status,
        qa_reason=qa.reason_code,
        gating=gating,
        estimates=estimates,
        combined=combined,
        combined_t_td=combined_t_td,
        combined_distance_m=combined_distance_m,
        lateral_offset_m=lateral_offset_m,
        naive_baseline=naive,
        no_touchdown=False,
        reason_code=reason_code,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Distance-error harness (Req 12.8 / 13.0)
# ---------------------------------------------------------------------------


def along_runway_truth_distance(runway: object, truth_lat: float, truth_lon: float) -> float:
    """Along-runway distance (m) of a truth touchdown lat/long (Req 12.10).

    A clock-independent geometric quantity: the truth touchdown point is
    projected onto the runway centerline directly, so the distance metric never
    depends on QAR-ADS-B clock alignment.
    """
    return project_to_runway(runway, float(truth_lat), float(truth_lon)).along_runway_distance_m


def _percentile(values: np.ndarray, q: float) -> float:
    """``np.percentile`` guarding the empty case (returns ``nan``)."""
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


def distance_error_summary(
    results: Sequence[StageRunResult],
    truths: Mapping[str, FlightTruth] | Sequence[FlightTruth],
) -> BaselineComparison:
    """Summarise the system-vs-baseline along-runway distance-error distribution.

    For every result with BOTH a provisional combined distance and an available
    naive baseline distance, and for which a truth is supplied, computes the
    along-runway truth distance from the truth lat/long (clock-independent) and
    the signed/absolute error of each method, then summarises RMSE, median and
    p95 absolute error plus the system-vs-baseline RMSE improvement (Req 12.8 /
    13.4). All distances are in metres.

    This is a PRELIMINARY synthetic characterisation of the cadence-limited error
    floor (Req 13.0); the full QAR validation harness is Task 22.

    Parameters
    ----------
    results:
        Per-flight :class:`StageRunResult` objects.
    truths:
        Either a ``{flight_id: FlightTruth}`` mapping or a sequence of
        :class:`FlightTruth` (indexed by ``flight_id``).

    Returns
    -------
    BaselineComparison
        The distance-error distribution and comparison (``n_flights`` is the
        number of like-for-like flights actually compared).
    """
    if isinstance(truths, Mapping):
        truth_by_id = dict(truths)
    else:
        truth_by_id = {t.flight_id: t for t in truths}

    system_errors: list[float] = []
    baseline_errors: list[float] = []

    for result in results:
        truth = truth_by_id.get(result.flight_id)
        if truth is None:
            continue
        if result.combined_distance_m is None:
            continue
        if not result.naive_baseline.available or result.naive_baseline.distance_m is None:
            continue
        truth_distance = along_runway_truth_distance(
            result.runway, truth.touchdown_lat, truth.touchdown_lon
        )
        system_errors.append(float(result.combined_distance_m) - truth_distance)
        baseline_errors.append(float(result.naive_baseline.distance_m) - truth_distance)

    sys_err = np.array(system_errors, dtype=float)
    base_err = np.array(baseline_errors, dtype=float)
    n = int(sys_err.size)

    def _rmse(e: np.ndarray) -> float:
        return float(np.sqrt(np.mean(e * e))) if e.size else float("nan")

    def _median_abs(e: np.ndarray) -> float:
        return float(np.median(np.abs(e))) if e.size else float("nan")

    def _median_signed(e: np.ndarray) -> float:
        return float(np.median(e)) if e.size else float("nan")

    system_rmse = _rmse(sys_err)
    baseline_rmse = _rmse(base_err)
    if baseline_rmse and np.isfinite(baseline_rmse) and baseline_rmse > 0.0:
        improvement = (baseline_rmse - system_rmse) / baseline_rmse * 100.0
    else:
        improvement = float("nan")

    return BaselineComparison(
        n_flights=n,
        system_rmse_m=system_rmse,
        system_median_abs_error_m=_median_abs(sys_err),
        system_p95_abs_error_m=_percentile(np.abs(sys_err), 95.0),
        system_median_signed_error_m=_median_signed(sys_err),
        baseline_rmse_m=baseline_rmse,
        baseline_median_abs_error_m=_median_abs(base_err),
        baseline_p95_abs_error_m=_percentile(np.abs(base_err), 95.0),
        baseline_median_signed_error_m=_median_signed(base_err),
        rmse_improvement_pct=improvement,
        cadence_limited_floor_m=system_rmse,
        system_errors_m=tuple(float(x) for x in sys_err),
        baseline_errors_m=tuple(float(x) for x in base_err),
    )
