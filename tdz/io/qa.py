"""Data-quality (QA) gates for ingest (Task 9.3).

Real ADS-B data carries duplicate timestamps, physically impossible samples,
gaps, and missing channels. This module cleans and gates a parsed
:class:`~tdz.models.FlightRecord` before estimation, returning a structured
:class:`QAResult` (cleaned record + diagnostics + an ``ok``/``no-estimate``
status), in line with the design's three-tier confidence model: QA rejections
are fatal-for-this-flight but are reported as a status + reason code, not raised
as exceptions.

Gates implemented
-----------------
* **Duplicate-timestamp dedup** (Req 9.3 / Property 8): samples whose timestamps
  fall within ``duplicate_timestamp_tolerance_s`` (0.1 s) of each other are
  collapsed to one, keeping the LAST-received in each group
  (:func:`deduplicate_by_timestamp`).

* **Kinematic plausibility gates** (Req 9.4 / Property 9): consecutive velocity
  samples implying longitudinal acceleration > ``max_longitudinal_accel_g`` g,
  lateral acceleration > ``max_lateral_accel_g`` g, or turn rate >
  ``max_turn_rate_deg_s`` are excluded, with counts and timestamps recorded
  (:func:`apply_kinematic_gates`). Removal is iterative so the surviving
  trajectory contains only physically plausible transitions.

* **Sufficiency rejection** (Req 9.5 / Req 1.6): emits a no-estimate with the
  matching :class:`~tdz.models.FailureReason` when groundspeed is entirely
  missing (``NO_GROUNDSPEED``), a continuous gap > ``max_gap_spanning_td_s``
  (15 s) spans the touchdown region (``GAP_SPANS_TOUCHDOWN``), more than
  ``max_excluded_fraction`` (0.5) of in-window samples were excluded
  (``EXCESSIVE_EXCLUSIONS``), or fewer than ``min_samples_in_window`` valid
  samples lie within ``window_half_width_s`` of the touchdown region
  (``INSUFFICIENT_SAMPLES``) (:func:`evaluate_sufficiency`).

* **Missing baro vertical rate tolerance** (Req 9.1 / Property 13): an all-NaN
  barometric-vertical-rate channel NEVER blocks ingest; it is surfaced as a
  diagnostic (``baro_vertical_rate_unavailable``) and the flight passes through.

Touchdown-window dependency
---------------------------
The coarse touchdown bracket is built in Task 10 (Module 1b), which does not yet
exist. The sufficiency gate therefore ACCEPTS the touchdown reference window as
a :class:`TouchdownWindow` parameter rather than computing it. Tests (and the
orchestrator's convenience path) use a simple stand-in -- the on-ground
transition time +/- ``window_half_width_s`` -- via
:meth:`TouchdownWindow.from_center`.

Formulas and constants
-----------------------
Longitudinal acceleration, lateral acceleration and turn rate between two
velocity samples ``(i-1, i)`` separated by ``dt = t[i] - t[i-1] > 0``:

* ``a_long = |gs[i] - gs[i-1]| * KNOTS_TO_MPS / dt``                 (m/s^2)
* ``turn_rate = |shortest_angle(track[i] - track[i-1])| / dt``      (deg/s)
* ``a_lat = (gs[i] * KNOTS_TO_MPS) * radians(turn_rate) ``           (m/s^2)
  (centripetal ``v * omega``, with ``omega`` the turn rate in rad/s)

Acceleration thresholds are expressed in g and converted with the standard
gravity constant :data:`STANDARD_GRAVITY_MPS2` = 9.80665 m/s^2: a sample is
excluded when ``a_long > max_longitudinal_accel_g * g`` or
``a_lat > max_lateral_accel_g * g`` or ``turn_rate > max_turn_rate_deg_s``. The
later sample of an offending pair is the one excluded.

Units convention
----------------
SI internally: groundspeed is converted from knots with
:data:`tdz.timebase.KNOTS_TO_MPS`; angles use radians for the lateral-accel
term. No conversion to feet/knots happens here (that is the output boundary).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from tdz.config.schema import QualityGatesConfig
from tdz.models import FailureReason, FlightRecord
from tdz.timebase import KNOTS_TO_MPS

__all__ = [
    "STANDARD_GRAVITY_MPS2",
    "TouchdownWindow",
    "DedupResult",
    "KinematicGateResult",
    "SufficiencyResult",
    "QADiagnostics",
    "QAResult",
    "deduplicate_by_timestamp",
    "apply_kinematic_gates",
    "evaluate_sufficiency",
    "run_qa",
]

#: Standard gravity (m/s^2), used to convert the g-denominated acceleration
#: gates to SI before comparison (Req 9.4).
STANDARD_GRAVITY_MPS2: Final[float] = 9.80665

# Gate labels (stable identifiers used in per-gate diagnostics).
_GATE_LONGITUDINAL: Final[str] = "longitudinal_accel"
_GATE_LATERAL: Final[str] = "lateral_accel"
_GATE_TURN: Final[str] = "turn_rate"


# ---------------------------------------------------------------------------
# Touchdown reference window (parameterized; the coarse bracket is Task 10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TouchdownWindow:
    """A coarse touchdown reference window ``[t_lo, t_hi]`` (epoch seconds).

    Supplied to the sufficiency gate so QA does not need the not-yet-built
    coarse bracket (Task 10). ``center`` defaults to the midpoint and is the
    point a gap must straddle to "span the touchdown" (Req 9.5).
    """

    t_lo: float
    t_hi: float
    center: float

    @classmethod
    def from_center(cls, center: float, half_width_s: float) -> "TouchdownWindow":
        """Build a symmetric window ``[center - hw, center + hw]``.

        ``half_width_s`` is ``QualityGatesConfig.window_half_width_s`` (30 s),
        i.e. the "+/- window_half_width_s of the touchdown region" of Req 9.5.
        """
        return cls(t_lo=center - half_width_s, t_hi=center + half_width_s, center=center)

    def contains(self, t: float) -> bool:
        """Whether time ``t`` lies within ``[t_lo, t_hi]`` (inclusive)."""
        return self.t_lo <= t <= self.t_hi


# ---------------------------------------------------------------------------
# Result value objects (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DedupResult:
    """Outcome of duplicate-timestamp deduplication.

    ``kept_indices`` are indices into the original array, time-sorted, one per
    unique-within-tolerance timestamp group (last-received retained).
    """

    kept_indices: np.ndarray
    n_input: int
    n_kept: int
    n_removed: int
    removed_timestamps: tuple[float, ...]


@dataclass(frozen=True)
class KinematicGateResult:
    """Outcome of the kinematic plausibility gates (Req 9.4 / Property 9).

    ``kept_indices`` index the surviving velocity samples (time-sorted); after
    removal every surviving consecutive pair is physically plausible.
    """

    kept_indices: np.ndarray
    excluded_indices: np.ndarray
    excluded_timestamps: tuple[float, ...]
    excluded_gate_labels: tuple[str, ...]
    excluded_count: int
    counts_by_gate: dict[str, int]
    gravity_mps2: float


@dataclass(frozen=True)
class SufficiencyResult:
    """Outcome of the data-sufficiency gate (Req 9.5 / Req 1.6).

    ``ok`` is ``True`` when enough data exists near the touchdown region;
    otherwise ``reason_code`` carries the triggering
    :class:`~tdz.models.FailureReason`.
    """

    ok: bool
    reason_code: Optional[FailureReason]
    n_valid_in_window: int
    excluded_fraction_in_window: float
    max_gap_spanning_s: float
    groundspeed_present: bool
    window: TouchdownWindow


@dataclass(frozen=True)
class QADiagnostics:
    """Diagnostic record summarizing all QA gates for one flight."""

    n_duplicates_removed: int
    duplicate_removed_timestamps: tuple[float, ...]
    excluded_sample_count: int
    excluded_sample_timestamps: tuple[float, ...]
    excluded_gate_labels: tuple[str, ...]
    counts_by_gate: dict[str, int]
    baro_vertical_rate_unavailable: bool
    unavailable_signals: tuple[str, ...]
    sufficiency: Optional[SufficiencyResult]


@dataclass(frozen=True)
class QAResult:
    """Structured QA outcome for one flight.

    ``status`` is ``"ok"`` or ``"no-estimate"``. A ``"no-estimate"`` carries a
    non-null ``reason_code`` (a :class:`~tdz.models.FailureReason`). The cleaned
    record (dedup + kinematic gating applied) is always returned for diagnostics
    even when the flight is rejected.
    """

    cleaned: FlightRecord
    status: str
    reason_code: Optional[FailureReason]
    diagnostics: QADiagnostics


# ---------------------------------------------------------------------------
# Duplicate-timestamp deduplication (Req 9.3 / Property 8)
# ---------------------------------------------------------------------------


def deduplicate_by_timestamp(
    times: np.ndarray, tolerance_s: float
) -> DedupResult:
    """Deduplicate samples whose timestamps fall within ``tolerance_s``.

    Samples are clustered by timestamp proximity: walking the time-sorted
    samples, each sample joins the current cluster while it is within
    ``tolerance_s`` of the cluster's first (earliest) member, otherwise it opens
    a new cluster. Each cluster collapses to a single retained sample: the
    LAST-received within the cluster, i.e. the one with the greatest *original*
    input index (input order is arrival order). For exact-duplicate groups this
    is exactly Property 8: the output has ``N - total_duplicates`` samples, one
    per unique timestamp, retaining the last-received.

    Parameters
    ----------
    times:
        Sample timestamps (epoch seconds); arbitrary input order.
    tolerance_s:
        Duplicate tolerance (``duplicate_timestamp_tolerance_s``, 0.1 s).

    Returns
    -------
    DedupResult
        ``kept_indices`` index the original array, time-sorted.
    """
    t = np.asarray(times, dtype=float)
    n = t.size
    if n == 0:
        return DedupResult(
            kept_indices=np.array([], dtype=int),
            n_input=0,
            n_kept=0,
            n_removed=0,
            removed_timestamps=(),
        )

    # Stable time-sort; ties keep original (arrival) order so "last-received"
    # is the greatest original index within a tied group.
    order = np.argsort(t, kind="stable")
    kept: list[int] = []
    removed: list[int] = []

    cluster_anchor_t = t[order[0]]
    cluster_members = [int(order[0])]

    def _flush(members: list[int]) -> None:
        # Retain the last-received (max original index); the rest are removed.
        keep = max(members)
        kept.append(keep)
        removed.extend(m for m in members if m != keep)

    for k in order[1:]:
        k = int(k)
        if t[k] - cluster_anchor_t <= tolerance_s:
            cluster_members.append(k)
        else:
            _flush(cluster_members)
            cluster_anchor_t = t[k]
            cluster_members = [k]
    _flush(cluster_members)

    kept_sorted = np.array(sorted(kept, key=lambda i: (t[i], i)), dtype=int)
    removed_sorted = sorted(removed, key=lambda i: (t[i], i))
    return DedupResult(
        kept_indices=kept_sorted,
        n_input=n,
        n_kept=kept_sorted.size,
        n_removed=len(removed_sorted),
        removed_timestamps=tuple(float(t[i]) for i in removed_sorted),
    )


# ---------------------------------------------------------------------------
# Kinematic plausibility gates (Req 9.4 / Property 9)
# ---------------------------------------------------------------------------


def _shortest_angle_deg(a0: float, a1: float) -> float:
    """Shortest signed angular difference ``a1 - a0`` in degrees, in (-180, 180]."""
    return ((a1 - a0 + 180.0) % 360.0) - 180.0


def _classify_transition(
    dt: float,
    gs0_kt: float,
    gs1_kt: float,
    track0_deg: float,
    track1_deg: float,
    gates: QualityGatesConfig,
    gravity: float,
) -> Optional[str]:
    """Return the gate label a transition violates, or ``None`` if plausible.

    Priority when multiple gates trip: longitudinal -> lateral -> turn. A
    transition with non-positive ``dt`` or NaN velocity is treated as plausible
    here (duplicates are removed earlier; NaN velocity is handled by the
    sufficiency/interpolation layers).
    """
    if not (dt > 0.0):
        return None
    if math.isnan(gs0_kt) or math.isnan(gs1_kt):
        return None

    a_long = abs(gs1_kt - gs0_kt) * KNOTS_TO_MPS / dt
    if a_long > gates.max_longitudinal_accel_g * gravity:
        return _GATE_LONGITUDINAL

    if not (math.isnan(track0_deg) or math.isnan(track1_deg)):
        turn_rate_deg_s = abs(_shortest_angle_deg(track0_deg, track1_deg)) / dt
        omega_rad_s = math.radians(turn_rate_deg_s)
        v_mps = gs1_kt * KNOTS_TO_MPS
        a_lat = v_mps * omega_rad_s
        if a_lat > gates.max_lateral_accel_g * gravity:
            return _GATE_LATERAL
        if turn_rate_deg_s > gates.max_turn_rate_deg_s:
            return _GATE_TURN

    return None


def apply_kinematic_gates(
    velocity_times: np.ndarray,
    groundspeeds_kt: np.ndarray,
    tracks_deg: np.ndarray,
    gates: QualityGatesConfig,
    *,
    gravity_mps2: float = STANDARD_GRAVITY_MPS2,
) -> KinematicGateResult:
    """Exclude velocity samples implying impossible kinematics (Req 9.4).

    Operates on the velocity timebase (groundspeed + track). For each pair of
    consecutive surviving samples the implied longitudinal acceleration, lateral
    acceleration and turn rate are computed (see module formulas); when a pair
    violates a gate the LATER sample is excluded. Removal is iterated until no
    surviving consecutive pair violates any gate, so the returned trajectory
    contains only physically plausible transitions (Property 9). The count and
    timestamps of excluded samples are recorded for the diagnostics output.

    Parameters
    ----------
    velocity_times, groundspeeds_kt, tracks_deg:
        The velocity timebase (epoch s), groundspeed (knots) and track (deg).
    gates:
        Thresholds from :class:`~tdz.config.schema.QualityGatesConfig`.
    gravity_mps2:
        Standard gravity used to convert the g-denominated thresholds; defaults
        to :data:`STANDARD_GRAVITY_MPS2`.

    Returns
    -------
    KinematicGateResult
        ``kept_indices`` (time-sorted, into the original arrays), the excluded
        indices/timestamps/gate-labels, the total excluded count and per-gate
        counts.
    """
    vt = np.asarray(velocity_times, dtype=float)
    gs = np.asarray(groundspeeds_kt, dtype=float)
    tr = np.asarray(tracks_deg, dtype=float)
    n = vt.size

    order = np.argsort(vt, kind="stable")
    kept = list(int(i) for i in order)

    excluded_idx: list[int] = []
    excluded_labels: list[str] = []

    while len(kept) >= 2:
        to_remove: list[int] = []
        remove_labels: list[str] = []
        for j in range(1, len(kept)):
            i0 = kept[j - 1]
            i1 = kept[j]
            dt = float(vt[i1] - vt[i0])
            label = _classify_transition(
                dt, gs[i0], gs[i1], tr[i0], tr[i1], gates, gravity_mps2
            )
            if label is not None:
                to_remove.append(i1)
                remove_labels.append(label)
        if not to_remove:
            break
        remove_set = set(to_remove)
        kept = [i for i in kept if i not in remove_set]
        excluded_idx.extend(to_remove)
        excluded_labels.extend(remove_labels)

    counts: dict[str, int] = {
        _GATE_LONGITUDINAL: 0,
        _GATE_LATERAL: 0,
        _GATE_TURN: 0,
    }
    for lab in excluded_labels:
        counts[lab] += 1

    # Sort excluded for stable, time-ordered diagnostics.
    paired = sorted(
        zip(excluded_idx, excluded_labels), key=lambda p: (vt[p[0]], p[0])
    )
    excluded_idx_sorted = [i for i, _ in paired]
    excluded_labels_sorted = [lab for _, lab in paired]

    return KinematicGateResult(
        kept_indices=np.array(sorted(kept, key=lambda i: (vt[i], i)), dtype=int),
        excluded_indices=np.array(excluded_idx_sorted, dtype=int),
        excluded_timestamps=tuple(float(vt[i]) for i in excluded_idx_sorted),
        excluded_gate_labels=tuple(excluded_labels_sorted),
        excluded_count=len(excluded_idx_sorted),
        counts_by_gate=counts,
        gravity_mps2=gravity_mps2,
    )


# ---------------------------------------------------------------------------
# Data-sufficiency gate (Req 9.5 / Req 1.6)
# ---------------------------------------------------------------------------


def evaluate_sufficiency(
    *,
    valid_times: np.ndarray,
    n_excluded_in_window: int,
    groundspeed_present: bool,
    window: TouchdownWindow,
    gates: QualityGatesConfig,
) -> SufficiencyResult:
    """Decide whether enough data exists near the touchdown region (Req 9.5).

    Precedence of the no-estimate reasons (most fundamental first): no
    groundspeed at all (``NO_GROUNDSPEED``) -> a continuous gap >
    ``max_gap_spanning_td_s`` straddling ``window.center``
    (``GAP_SPANS_TOUCHDOWN``) -> more than ``max_excluded_fraction`` of in-window
    samples excluded (``EXCESSIVE_EXCLUSIONS``) -> fewer than
    ``min_samples_in_window`` valid in-window samples (``INSUFFICIENT_SAMPLES``).

    Parameters
    ----------
    valid_times:
        Timestamps of the VALID (post-gate) samples to assess (typically the
        velocity/groundspeed timebase). Need not be pre-filtered to the window.
    n_excluded_in_window:
        Count of samples excluded by the kinematic gates whose timestamps fall
        inside ``[window.t_lo, window.t_hi]``.
    groundspeed_present:
        ``False`` when groundspeed is entirely missing/empty (-> NO_GROUNDSPEED).
    window:
        The touchdown reference window (parameterized; see module docstring).
    gates:
        Thresholds from :class:`~tdz.config.schema.QualityGatesConfig`.

    Returns
    -------
    SufficiencyResult
        ``ok`` plus the metrics used in the decision.
    """
    vt = np.sort(np.asarray(valid_times, dtype=float))
    in_window = vt[(vt >= window.t_lo) & (vt <= window.t_hi)]
    n_valid = int(in_window.size)

    # Largest gap that straddles the touchdown center (use all valid samples so
    # a gap bracketing the window is detected even if endpoints sit outside it).
    max_gap_spanning = 0.0
    if vt.size >= 2:
        gaps = np.diff(vt)
        for k, g in enumerate(gaps):
            if vt[k] <= window.center <= vt[k + 1] and g > max_gap_spanning:
                max_gap_spanning = float(g)
    # A gap can also span the center when there are valid samples on only one
    # side (the "gap" extends from the last sample before center to t_hi, or
    # from t_lo to the first sample after center). This is covered by the
    # n_valid check below; explicit gap detection here requires bracketing
    # samples on both sides, matching "a continuous gap that spans t_td".

    denom = n_valid + n_excluded_in_window
    excluded_fraction = (n_excluded_in_window / denom) if denom > 0 else 0.0

    reason: Optional[FailureReason] = None
    if not groundspeed_present:
        reason = FailureReason.NO_GROUNDSPEED
    elif max_gap_spanning > gates.max_gap_spanning_td_s:
        reason = FailureReason.GAP_SPANS_TOUCHDOWN
    elif excluded_fraction > gates.max_excluded_fraction:
        reason = FailureReason.EXCESSIVE_EXCLUSIONS
    elif n_valid < gates.min_samples_in_window:
        reason = FailureReason.INSUFFICIENT_SAMPLES

    return SufficiencyResult(
        ok=reason is None,
        reason_code=reason,
        n_valid_in_window=n_valid,
        excluded_fraction_in_window=excluded_fraction,
        max_gap_spanning_s=max_gap_spanning,
        groundspeed_present=groundspeed_present,
        window=window,
    )


# ---------------------------------------------------------------------------
# Flight-level orchestration
# ---------------------------------------------------------------------------


def _index_subset(arr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return ``arr[idx]`` guarding empty arrays / index arrays."""
    a = np.asarray(arr)
    if a.size == 0 or idx.size == 0:
        return a[:0] if idx.size == 0 else a
    return a[idx]


def _groundspeed_present(groundspeeds_kt: np.ndarray) -> bool:
    """Whether any finite groundspeed exists (else NO_GROUNDSPEED)."""
    gs = np.asarray(groundspeeds_kt, dtype=float)
    return bool(gs.size) and bool(np.any(~np.isnan(gs)))


def run_qa(
    record: FlightRecord,
    gates: QualityGatesConfig,
    *,
    touchdown_window: Optional[TouchdownWindow] = None,
) -> QAResult:
    """Run the full QA pipeline on a parsed flight (Req 9.1, 9.3, 9.4, 9.5).

    Steps, in order: (1) deduplicate the position and velocity streams; for a
    co-timed source (``position_times == velocity_times``) the two streams are
    deduplicated together so rows stay aligned, otherwise independently.
    (2) Apply the kinematic gates to the velocity stream; for a co-timed source
    the excluded rows are dropped from the position arrays too. (3) Surface the
    missing-baro-vertical-rate diagnostic WITHOUT rejecting the flight
    (Req 9.1 / Property 13). (4) If a ``touchdown_window`` is supplied (or can be
    derived from the on-ground transition time), evaluate the sufficiency gate
    and set the no-estimate status/reason accordingly (Req 9.5).

    The cleaned :class:`FlightRecord` (dedup + gating applied) is always
    returned, even on rejection, for diagnostics.

    Parameters
    ----------
    record:
        The parsed flight (from :mod:`tdz.io.ingest`).
    gates:
        Thresholds from :class:`~tdz.config.schema.QualityGatesConfig`.
    touchdown_window:
        The coarse touchdown reference window (parameterized; Task 10 will
        supply it). When ``None``, a stand-in is derived from
        ``record.on_ground_transition_time`` +/- ``window_half_width_s``; if that
        is also unavailable the sufficiency gate is skipped (status ``ok``).

    Returns
    -------
    QAResult
        Cleaned record, ``ok``/``no-estimate`` status, reason code, diagnostics.
    """
    co_timed = (
        record.position_times.size == record.velocity_times.size
        and bool(np.array_equal(record.position_times, record.velocity_times))
    )

    # --- Step 1: dedup ----------------------------------------------------
    if co_timed:
        dedup = deduplicate_by_timestamp(
            record.velocity_times, gates.duplicate_timestamp_tolerance_s
        )
        pos_keep = dedup.kept_indices
        vel_keep = dedup.kept_indices
        n_dups = dedup.n_removed
        dup_ts = dedup.removed_timestamps
    else:
        pos_dedup = deduplicate_by_timestamp(
            record.position_times, gates.duplicate_timestamp_tolerance_s
        )
        vel_dedup = deduplicate_by_timestamp(
            record.velocity_times, gates.duplicate_timestamp_tolerance_s
        )
        pos_keep = pos_dedup.kept_indices
        vel_keep = vel_dedup.kept_indices
        n_dups = pos_dedup.n_removed + vel_dedup.n_removed
        dup_ts = tuple(
            sorted(pos_dedup.removed_timestamps + vel_dedup.removed_timestamps)
        )

    position_times = _index_subset(record.position_times, pos_keep)
    latitudes = _index_subset(record.latitudes, pos_keep)
    longitudes = _index_subset(record.longitudes, pos_keep)
    geometric_altitudes = _index_subset(record.geometric_altitudes, pos_keep)
    barometric_altitudes = _index_subset(record.barometric_altitudes, pos_keep)
    on_ground_flags = _index_subset(record.on_ground_flags, pos_keep)

    velocity_times = _index_subset(record.velocity_times, vel_keep)
    groundspeeds = _index_subset(record.groundspeeds, vel_keep)
    tracks = _index_subset(record.tracks, vel_keep)
    baro_vertical_rates = _index_subset(record.baro_vertical_rates, vel_keep)

    # --- Step 2: kinematic gates (on the velocity stream) -----------------
    kin = apply_kinematic_gates(velocity_times, groundspeeds, tracks, gates)
    vel_gate_keep = kin.kept_indices

    velocity_times_g = _index_subset(velocity_times, vel_gate_keep)
    groundspeeds_g = _index_subset(groundspeeds, vel_gate_keep)
    tracks_g = _index_subset(tracks, vel_gate_keep)
    baro_vertical_rates_g = _index_subset(baro_vertical_rates, vel_gate_keep)

    if co_timed:
        # Rows are aligned: drop the same indices from the position arrays.
        position_times = _index_subset(position_times, vel_gate_keep)
        latitudes = _index_subset(latitudes, vel_gate_keep)
        longitudes = _index_subset(longitudes, vel_gate_keep)
        geometric_altitudes = _index_subset(geometric_altitudes, vel_gate_keep)
        barometric_altitudes = _index_subset(barometric_altitudes, vel_gate_keep)
        on_ground_flags = _index_subset(on_ground_flags, vel_gate_keep)
        position_times_out = position_times
    else:
        position_times_out = position_times

    # --- Step 3: missing-signal diagnostics (never a rejection) -----------
    baro_vr_unavailable = (
        baro_vertical_rates_g.size == 0
        or bool(np.all(np.isnan(baro_vertical_rates_g)))
    )
    unavailable: list[str] = []
    if baro_vr_unavailable:
        unavailable.append("baro_vertical_rate")
    if geometric_altitudes.size == 0 or bool(np.all(np.isnan(geometric_altitudes))):
        unavailable.append("geometric_altitude")

    cleaned = FlightRecord(
        flight_id=record.flight_id,
        aircraft_type=record.aircraft_type,
        ads_b_source=record.ads_b_source,
        position_times=position_times_out,
        velocity_times=velocity_times_g,
        latitudes=latitudes,
        longitudes=longitudes,
        geometric_altitudes=geometric_altitudes,
        barometric_altitudes=barometric_altitudes,
        groundspeeds=groundspeeds_g,
        tracks=tracks_g,
        baro_vertical_rates=baro_vertical_rates_g,
        on_ground_flags=on_ground_flags,
        on_ground_transition_time=record.on_ground_transition_time,
        runway=record.runway,
    )

    # --- Step 4: sufficiency gate (uses the parameterized window) ---------
    window = touchdown_window
    if window is None and record.on_ground_transition_time is not None:
        window = TouchdownWindow.from_center(
            record.on_ground_transition_time, gates.window_half_width_s
        )

    sufficiency: Optional[SufficiencyResult] = None
    status = "ok"
    reason_code: Optional[FailureReason] = None
    if window is not None:
        gs_present = _groundspeed_present(record.groundspeeds)
        n_excluded_in_window = int(
            sum(1 for t in kin.excluded_timestamps if window.contains(t))
        )
        sufficiency = evaluate_sufficiency(
            valid_times=velocity_times_g,
            n_excluded_in_window=n_excluded_in_window,
            groundspeed_present=gs_present,
            window=window,
            gates=gates,
        )
        if not sufficiency.ok:
            status = "no-estimate"
            reason_code = sufficiency.reason_code

    diagnostics = QADiagnostics(
        n_duplicates_removed=n_dups,
        duplicate_removed_timestamps=tuple(dup_ts),
        excluded_sample_count=kin.excluded_count,
        excluded_sample_timestamps=kin.excluded_timestamps,
        excluded_gate_labels=kin.excluded_gate_labels,
        counts_by_gate=dict(kin.counts_by_gate),
        baro_vertical_rate_unavailable=baro_vr_unavailable,
        unavailable_signals=tuple(unavailable),
        sufficiency=sufficiency,
    )

    return QAResult(
        cleaned=cleaned,
        status=status,
        reason_code=reason_code,
        diagnostics=diagnostics,
    )
