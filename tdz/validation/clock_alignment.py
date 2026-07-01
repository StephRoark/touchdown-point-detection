"""QAR <-> ADS-B clock alignment for time-domain truth preparation (Task 21).

This module estimates, per matched flight, the clock offset between the QAR
ground-truth clock and the ADS-B clock, so a systematic time bias does not
corrupt time-domain training labels or the time-error metric (Requirement 19).

Design (see design "Clock Alignment (Truth Preparation)")
---------------------------------------------------------
* The offset is estimated by **cross-correlating an overlapping kinematic
  series** common to both streams -- groundspeed and/or along-track position
  over the approach and rollout -- and taking the lag that maximizes alignment
  (Req 19.1). The touchdown event itself is **never** used to align, since that
  would be circular with the quantity being estimated.
* Within-flight **drift** (a lag that varies across the trajectory, not a single
  constant offset) is detected by estimating the lag independently over
  contiguous sub-windows and measuring the spread; flights whose drift exceeds
  ``clock_drift_max_s`` are flagged (Req 19.2).
* Clock alignment is applied to QAR timestamps for **time-domain** labels/metrics
  ONLY (Req 19.3, 19.4). The along-runway distance truth is derived geometrically
  from the QAR touchdown lat/long and stays clock-independent -- this module
  never touches it.
* The corpus offset **distribution** (median, standard deviation, 95th-percentile
  absolute) is reported as a data-quality diagnostic (Req 19.5).
* Flights whose estimated offset exceeds ``clock_offset_max_s`` (default 2 s), or
  whose offset cannot be reliably estimated (insufficient overlap / weak
  correlation), or that exhibit excessive drift, are **excluded from time-domain
  training/validation** and recorded in the flagged-flights report, tagged
  :attr:`~tdz.models.FailureReason.CLOCK_OFFSET_EXCEEDED`. They are **retained**
  for clock-independent distance validation (Req 19.6, 19.7).

Sign convention
---------------
The estimated offset is ``QAR - ADS-B`` (seconds): a QAR sample stamped at time
``t`` corresponds to the same physical instant as an ADS-B sample stamped at
``t - offset``. Aligning QAR onto the ADS-B clock therefore **subtracts** the
offset (see :func:`apply_offset_to_qar`).

Units: all times are seconds, all series values are SI (m/s for groundspeed, m
for along-track position). Thresholds/offsets are seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Optional

import numpy as np

from tdz.config.schema import ValidationConfig
from tdz.models import FailureReason, QARTruthRecord

__all__ = [
    "QUALITY_GOOD",
    "QUALITY_DEGRADED",
    "QUALITY_FAILED",
    "KinematicSeries",
    "ClockOffsetResult",
    "OffsetDistribution",
    "ClockAlignmentReport",
    "estimate_clock_offset",
    "align_corpus",
    "apply_offset_to_qar",
]

#: The three offset-quality classes carried on
#: :attr:`~tdz.models.QARTruthRecord.clock_offset_quality`.
QUALITY_GOOD: Final[str] = "good"
QUALITY_DEGRADED: Final[str] = "degraded"
QUALITY_FAILED: Final[str] = "failed"


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass
class KinematicSeries:
    """An overlapping kinematic time series from one clock/stream.

    ``times`` are epoch seconds on that stream's own clock; ``values`` are the
    kinematic quantity sampled at those times (SI: m/s for groundspeed, m for
    along-track position). The two series passed to :func:`estimate_clock_offset`
    must describe the *same* physical quantity (e.g. both groundspeed) so their
    shapes align under a pure time shift.
    """

    times: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.times.shape != self.values.shape:
            raise ValueError(
                "KinematicSeries times and values must have the same shape; "
                f"got {self.times.shape} and {self.values.shape}"
            )
        if self.times.ndim != 1:
            raise ValueError("KinematicSeries expects 1-D times/values")


@dataclass
class ClockOffsetResult:
    """Per-flight clock-offset estimate and time-domain exclusion decision.

    ``offset_s`` is the estimated ``QAR - ADS-B`` offset (seconds), or ``None``
    when estimation failed. ``quality`` is one of ``"good" | "degraded" |
    "failed"``. ``excluded_time_domain`` is ``True`` when the flight must be
    dropped from time-domain training/validation (over-threshold offset,
    excessive drift, or a failed estimate); such flights carry
    ``reason_code == CLOCK_OFFSET_EXCEEDED`` and appear in the flagged report,
    yet remain usable for clock-independent distance validation.
    """

    flight_id: str
    offset_s: Optional[float]
    quality: str
    drift_s: Optional[float]
    drift_exceeded: bool
    overlap_s: float
    peak_correlation: float
    excluded_time_domain: bool
    reason_code: Optional[str]
    diagnostics: dict = field(default_factory=dict)


@dataclass
class OffsetDistribution:
    """Corpus offset distribution reported as a data-quality diagnostic (Req 19.5)."""

    n: int
    median_s: float
    sd_s: float
    p95_abs_s: float


@dataclass
class ClockAlignmentReport:
    """Corpus-level clock-alignment outcome.

    ``results`` holds one :class:`ClockOffsetResult` per input flight (input
    order preserved). ``distribution`` summarizes the reliably-estimated offsets.
    ``flagged_flights`` is the subset of ``results`` excluded from the time
    domain (the flagged-flights report of Req 19.6/19.7).
    """

    results: list[ClockOffsetResult]
    distribution: OffsetDistribution
    flagged_flights: list[ClockOffsetResult]


# ---------------------------------------------------------------------------
# Cross-correlation lag estimation
# ---------------------------------------------------------------------------


def _resample_common_grid(
    series_a: KinematicSeries,
    series_b: KinematicSeries,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample both series onto a shared uniform grid over their overlap.

    Returns ``(grid_a, grid_b, overlap_s)`` where ``grid_a``/``grid_b`` are the
    two series linearly interpolated onto ``arange(lo, hi, dt)`` and ``overlap_s``
    is ``hi - lo``. When the time supports do not overlap, ``overlap_s`` is 0 and
    the grids are empty.
    """
    lo = max(float(series_a.times.min()), float(series_b.times.min()))
    hi = min(float(series_a.times.max()), float(series_b.times.max()))
    overlap_s = hi - lo
    if not math.isfinite(overlap_s) or overlap_s <= 0.0:
        empty = np.array([], dtype=float)
        return empty, empty, max(overlap_s, 0.0)

    grid = np.arange(lo, hi + dt * 0.5, dt)
    # np.interp requires increasing x; sort defensively (series are usually sorted).
    order_a = np.argsort(series_a.times)
    order_b = np.argsort(series_b.times)
    grid_a = np.interp(grid, series_a.times[order_a], series_a.values[order_a])
    grid_b = np.interp(grid, series_b.times[order_b], series_b.values[order_b])
    return grid_a, grid_b, overlap_s


def _best_lag(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    dt: float,
    max_lag_s: float,
) -> tuple[Optional[float], float]:
    """Return ``(offset_s, peak_correlation)`` maximizing normalized correlation.

    ``grid_a`` is the QAR-derived series and ``grid_b`` the ADS-B series, both on
    the same uniform grid (step ``dt``). For an integer lag ``k`` we compare
    ``grid_a[i]`` with ``grid_b[i - k]``; the maximizing ``k`` gives
    ``offset_s = k * dt`` (the ``QAR - ADS-B`` shift). The peak is refined to
    sub-grid resolution by a parabolic fit around the best integer lag. Returns
    ``(None, corr)`` when the signals are degenerate (zero variance) or too short.

    Each candidate lag is scored by the **normalized** cross-correlation
    (Pearson coefficient) over its overlapping segment, so the peak is not biased
    toward lag 0 by the larger term count at full overlap.
    """
    n = grid_a.size
    if n < 2:
        return None, 0.0

    # Whole-series variance guard: a flat series carries no timing information.
    if float(np.std(grid_a)) == 0.0 or float(np.std(grid_b)) == 0.0:
        return None, 0.0

    max_k = int(math.floor(max_lag_s / dt))
    max_k = max(1, min(max_k, n - 1))
    # Require each lag's overlap to retain at least half the grid (and >= 3
    # samples) so short, trivially-correlated tails cannot win.
    min_seg = max(3, n // 2)

    lags = np.arange(-max_k, max_k + 1)
    corrs = np.full(lags.size, -np.inf, dtype=float)
    for idx, k in enumerate(lags):
        # a[i] vs b[i - k]  ->  overlap region depends on sign of k.
        if k >= 0:
            seg_a = grid_a[k:]
            seg_b = grid_b[: n - k]
        else:
            seg_a = grid_a[: n + k]
            seg_b = grid_b[-k:]
        if seg_a.size < min_seg:
            continue
        sa = seg_a - seg_a.mean()
        sb = seg_b - seg_b.mean()
        na = float(np.linalg.norm(sa))
        nb = float(np.linalg.norm(sb))
        if na == 0.0 or nb == 0.0:
            continue
        corrs[idx] = float(np.dot(sa, sb) / (na * nb))

    if not np.any(np.isfinite(corrs)):
        return None, 0.0

    best_idx = int(np.argmax(corrs))
    peak = float(corrs[best_idx])
    best_k = float(lags[best_idx])

    # Parabolic sub-grid refinement around the discrete peak (when interior and
    # the neighbouring correlations form a concave-down triple).
    if 0 < best_idx < lags.size - 1:
        c0 = corrs[best_idx - 1]
        c1 = corrs[best_idx]
        c2 = corrs[best_idx + 1]
        denom = c0 - 2.0 * c1 + c2
        if math.isfinite(denom) and denom < 0.0:
            delta = 0.5 * (c0 - c2) / denom
            if -1.0 < delta < 1.0:
                best_k += delta

    return float(best_k * dt), peak


def _segment_lags(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    dt: float,
    max_lag_s: float,
    n_segments: int,
) -> list[float]:
    """Estimate the lag independently over ``n_segments`` contiguous sub-windows.

    Used for drift detection: a within-flight clock drift shows up as the
    per-segment lag varying across the trajectory. Segments whose signal is
    degenerate (no variance) are skipped.
    """
    n = grid_a.size
    if n_segments < 2 or n < n_segments * 2:
        return []

    bounds = np.linspace(0, n, n_segments + 1).astype(int)
    lags: list[float] = []
    for i in range(n_segments):
        s, e = bounds[i], bounds[i + 1]
        if e - s < 2:
            continue
        lag, _ = _best_lag(grid_a[s:e], grid_b[s:e], dt, max_lag_s)
        if lag is not None:
            lags.append(lag)
    return lags


# ---------------------------------------------------------------------------
# Per-flight estimation
# ---------------------------------------------------------------------------


def estimate_clock_offset(
    flight_id: str,
    qar_series: KinematicSeries,
    adsb_series: KinematicSeries,
    config: ValidationConfig,
) -> ClockOffsetResult:
    """Estimate the QAR<->ADS-B clock offset for one flight via cross-correlation.

    Parameters
    ----------
    flight_id:
        Identifier carried through to the result and the flagged report.
    qar_series, adsb_series:
        Overlapping kinematic series (same physical quantity) on the QAR and
        ADS-B clocks respectively (Req 19.1). Touchdown is deliberately not an
        input -- alignment uses the approach+rollout kinematics only.
    config:
        Validation config carrying ``clock_offset_max_s`` (exclusion threshold),
        ``clock_drift_max_s`` (drift bound), ``clock_xcorr_resample_dt_s``,
        ``clock_max_lag_search_s``, ``clock_min_overlap_s``,
        ``clock_min_peak_correlation`` and ``clock_drift_segments``.

    Returns
    -------
    ClockOffsetResult
        Estimated offset, quality, drift, and the time-domain exclusion decision.
    """
    dt = config.clock_xcorr_resample_dt_s
    grid_a, grid_b, overlap_s = _resample_common_grid(qar_series, adsb_series, dt)

    diagnostics: dict = {
        "resample_dt_s": dt,
        "max_lag_search_s": config.clock_max_lag_search_s,
        "min_overlap_s": config.clock_min_overlap_s,
        "n_grid_samples": int(grid_a.size),
    }

    # --- Insufficient overlap -> failed estimate (Req 19.7) -----------------
    if overlap_s < config.clock_min_overlap_s or grid_a.size < 2:
        diagnostics["failure"] = "insufficient_overlap"
        return ClockOffsetResult(
            flight_id=flight_id,
            offset_s=None,
            quality=QUALITY_FAILED,
            drift_s=None,
            drift_exceeded=False,
            overlap_s=overlap_s,
            peak_correlation=0.0,
            excluded_time_domain=True,
            reason_code=FailureReason.CLOCK_OFFSET_EXCEEDED.value,
            diagnostics=diagnostics,
        )

    offset_s, peak = _best_lag(grid_a, grid_b, dt, config.clock_max_lag_search_s)
    diagnostics["peak_correlation"] = peak

    # --- Weak/degenerate correlation -> failed estimate (Req 19.7) ----------
    if offset_s is None or peak < config.clock_min_peak_correlation:
        diagnostics["failure"] = (
            "degenerate_series" if offset_s is None else "weak_correlation"
        )
        return ClockOffsetResult(
            flight_id=flight_id,
            offset_s=None,
            quality=QUALITY_FAILED,
            drift_s=None,
            drift_exceeded=False,
            overlap_s=overlap_s,
            peak_correlation=peak,
            excluded_time_domain=True,
            reason_code=FailureReason.CLOCK_OFFSET_EXCEEDED.value,
            diagnostics=diagnostics,
        )

    # --- Within-flight drift detection (Req 19.2) ---------------------------
    seg_lags = _segment_lags(
        grid_a, grid_b, dt, config.clock_max_lag_search_s, config.clock_drift_segments
    )
    drift_s: Optional[float] = None
    if len(seg_lags) >= 2:
        drift_s = float(max(seg_lags) - min(seg_lags))
        diagnostics["segment_lags_s"] = seg_lags
    drift_exceeded = drift_s is not None and drift_s > config.clock_drift_max_s

    # --- Threshold-based exclusion (Req 19.6) -------------------------------
    offset_exceeded = bool(abs(offset_s) > config.clock_offset_max_s)
    excluded = bool(offset_exceeded or drift_exceeded)

    if excluded:
        quality = QUALITY_FAILED if offset_exceeded else QUALITY_DEGRADED
        reason_code = FailureReason.CLOCK_OFFSET_EXCEEDED.value
    elif drift_s is not None and drift_s > 0.5 * config.clock_drift_max_s:
        # Estimate is usable but shows appreciable (sub-threshold) drift.
        quality = QUALITY_DEGRADED
        reason_code = None
    else:
        quality = QUALITY_GOOD
        reason_code = None

    diagnostics["offset_exceeded"] = offset_exceeded
    diagnostics["drift_exceeded"] = drift_exceeded

    return ClockOffsetResult(
        flight_id=flight_id,
        offset_s=float(offset_s),
        quality=quality,
        drift_s=drift_s,
        drift_exceeded=drift_exceeded,
        overlap_s=overlap_s,
        peak_correlation=peak,
        excluded_time_domain=excluded,
        reason_code=reason_code,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Corpus-level alignment
# ---------------------------------------------------------------------------


def _offset_distribution(results: list[ClockOffsetResult]) -> OffsetDistribution:
    """Summarize reliably-estimated offsets (Req 19.5).

    Median and standard deviation are of the signed offsets; the 95th percentile
    is of the absolute offsets. Failed estimates (no offset) are omitted.
    """
    offsets = [r.offset_s for r in results if r.offset_s is not None]
    if not offsets:
        return OffsetDistribution(n=0, median_s=0.0, sd_s=0.0, p95_abs_s=0.0)
    arr = np.asarray(offsets, dtype=float)
    return OffsetDistribution(
        n=int(arr.size),
        median_s=float(np.median(arr)),
        sd_s=float(np.std(arr)),
        p95_abs_s=float(np.percentile(np.abs(arr), 95.0)),
    )


def align_corpus(
    series_by_flight: dict[str, tuple[KinematicSeries, KinematicSeries]],
    config: ValidationConfig,
) -> ClockAlignmentReport:
    """Estimate offsets for a corpus and build the flagged-flights report.

    Parameters
    ----------
    series_by_flight:
        Mapping ``flight_id -> (qar_series, adsb_series)`` of the overlapping
        kinematic series for each matched flight.
    config:
        Validation config (see :func:`estimate_clock_offset`).

    Returns
    -------
    ClockAlignmentReport
        Per-flight results, the offset distribution diagnostic (Req 19.5), and
        the flagged-flights report (the time-domain-excluded subset; Req 19.6/7).
    """
    results = [
        estimate_clock_offset(flight_id, qar, adsb, config)
        for flight_id, (qar, adsb) in series_by_flight.items()
    ]
    distribution = _offset_distribution(results)
    flagged = [r for r in results if r.excluded_time_domain]
    return ClockAlignmentReport(
        results=results, distribution=distribution, flagged_flights=flagged
    )


# ---------------------------------------------------------------------------
# Applying the offset to QAR truth (time-domain only)
# ---------------------------------------------------------------------------


def apply_offset_to_qar(
    record: QARTruthRecord,
    result: ClockOffsetResult,
) -> QARTruthRecord:
    """Return a QAR record aligned onto the ADS-B clock for time-domain use.

    The offset is ``QAR - ADS-B``, so aligning subtracts it from the QAR
    touchdown time (Req 19.4). The estimate and its quality are recorded on the
    returned record. Only :attr:`~tdz.models.QARTruthRecord.touchdown_time_qar`
    is shifted; the touchdown lat/long (which drive the clock-independent
    distance truth) are left untouched (Req 19.3).

    A record whose offset estimation failed (``offset_s is None``) is returned
    with its touchdown time unchanged and quality ``"failed"`` -- it must be
    excluded from the time domain by the caller (it already carries
    ``excluded_time_domain``), never silently used.
    """
    if result.offset_s is None:
        return QARTruthRecord(
            flight_id=record.flight_id,
            touchdown_time_qar=record.touchdown_time_qar,
            touchdown_lat=record.touchdown_lat,
            touchdown_lon=record.touchdown_lon,
            clock_offset_estimate=None,
            clock_offset_quality=QUALITY_FAILED,
            aircraft_type=record.aircraft_type,
            runway_id=record.runway_id,
            airport_id=record.airport_id,
            tail_number=record.tail_number,
        )

    return QARTruthRecord(
        flight_id=record.flight_id,
        touchdown_time_qar=record.touchdown_time_qar - result.offset_s,
        touchdown_lat=record.touchdown_lat,
        touchdown_lon=record.touchdown_lon,
        clock_offset_estimate=result.offset_s,
        clock_offset_quality=result.quality,
        aircraft_type=record.aircraft_type,
        runway_id=record.runway_id,
        airport_id=record.airport_id,
        tail_number=record.tail_number,
    )
