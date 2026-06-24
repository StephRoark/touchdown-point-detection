"""Trajectory classification: landing vs go-around vs touch-and-go (Task 10.1).

Runs before any estimator (design Module 1b, "Trajectory classified before
estimating"). Only a *completed landing* produces a touchdown estimate; a
go-around has no touchdown and short-circuits to a no-touchdown result; a
touch-and-go is tagged (reported-or-suppressed per policy). A bounce (more than
one main-gear contact during a single landing) is a completed landing whose
touchdown is the **first** contact -- the bracket (Task 10.2) anchors there so
the second-pass estimators never average across the bounce (Req 21.4 /
Property 21).

Classification signature
-------------------------
The discriminating signal is the post-contact behaviour of the vertical profile
together with ground contact:

* **completed-landing** -- the trajectory descends to the runway, makes contact,
  and ends *in sustained ground contact* without climbing away (continued
  ground roll / deceleration). Bounces (a brief re-rise below
  :data:`CLIMB_OUT_HEIGHT_M`) are still completed landings.
* **go-around** -- an approach/descent that **never makes contact** and climbs
  out again. Emits :attr:`~tdz.models.FailureReason.GO_AROUND` (Req 21.2).
* **touch-and-go** -- a brief contact **followed by a climb-out** above
  :data:`CLIMB_OUT_HEIGHT_M` without sustained ground roll. Tagged with
  :attr:`~tdz.models.FailureReason.TOUCH_AND_GO`; whether it still yields a
  touchdown bracket is governed by ``touch_and_go_policy`` (Req 21.3).

Ground contact is detected flag-independently: a position sample is "in contact"
when the on-ground flag is set **or** the geometric altitude has descended to
within :data:`CONTACT_HEIGHT_M` of the runway threshold elevation (resolved to
HAE so the comparison is datum-consistent -- :func:`resolve_threshold_elevation_hae`).
When the datum cannot be resolved (or no geometric altitude is available), the
classifier degrades to a *relative* altitude test (height above the trajectory
minimum) combined with the on-ground flag, so a trajectory is still classifiable
without an absolute vertical reference (see ``datum_resolved`` in the result).

Multiple landings (Req 21.5)
----------------------------
The system does **not** assume exactly one landing per input trajectory. The
upstream-segmentation assumption is that each input record contains a single
landing; where it does not (e.g. training circuits), this module *detects* the
condition: two contact groups separated by a full climb-out above
:data:`CLIMB_OUT_HEIGHT_M` count as separate landings, surfaced via
``n_landings`` / ``multiple_landings``. The reported first-contact time always
targets the **first** landing's first contact.

Threshold constants
-------------------
The module-level :data:`CONTACT_HEIGHT_M`, :data:`CLIMB_OUT_HEIGHT_M`,
:data:`SUSTAINED_GROUND_ROLL_S` and :data:`GROUND_ROLL_DECEL_DELTA_MPS` are
estimation-affecting thresholds given documented defaults here; they SHOULD
migrate to a dedicated configuration block later (cf. Req 20.2). Every public
entry point accepts them as keyword parameters so they can be externalised
without code changes in the meantime.

Units convention
----------------
SI internally: heights in metres (HAE), times in epoch seconds, groundspeed
converted from knots with :data:`tdz.timebase.KNOTS_TO_MPS`. No conversion to
feet/knots happens here (that is the output boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from tdz.geo.datum import resolve_threshold_elevation_hae
from tdz.geo.errors import DatumUnresolvedError
from tdz.models import FailureReason, FlightRecord
from tdz.timebase import KNOTS_TO_MPS

__all__ = [
    "CONTACT_HEIGHT_M",
    "CLIMB_OUT_HEIGHT_M",
    "SUSTAINED_GROUND_ROLL_S",
    "GROUND_ROLL_DECEL_DELTA_MPS",
    "TRAJECTORY_TYPES",
    "TRAJECTORY_COMPLETED_LANDING",
    "TRAJECTORY_GO_AROUND",
    "TRAJECTORY_TOUCH_AND_GO",
    "ContactSegment",
    "TrajectoryClassification",
    "classify_trajectory",
    "classification_confusion_matrix",
]

# ---------------------------------------------------------------------------
# Threshold constants (documented defaults; SHOULD migrate to config -- Req 20.2)
# ---------------------------------------------------------------------------

#: Height (m) above the runway threshold elevation (HAE) below which a position
#: sample is treated as in ground contact. A few metres absorbs antenna height
#: above the main gear and geometric-altitude noise near the surface. Migrate to
#: config. Externalisable via the ``contact_height_m`` parameter.
CONTACT_HEIGHT_M: Final[float] = 5.0

#: Height (m) above the runway threshold elevation (~200 ft) that a climb after
#: contact must exceed to count as a re-takeoff (go-around / touch-and-go) rather
#: than a bounce. A bounce re-rises below this and remains a completed landing.
#: Migrate to config. Externalisable via the ``climb_out_height_m`` parameter.
CLIMB_OUT_HEIGHT_M: Final[float] = 60.0

#: Minimum duration (s) of continued ground contact for the contact to count as
#: a *sustained* ground roll (corroborates a completed landing vs a brief
#: touch-and-go contact). Migrate to config. Externalisable via the
#: ``sustained_ground_roll_s`` parameter.
SUSTAINED_GROUND_ROLL_S: Final[float] = 8.0

#: Minimum groundspeed drop (m/s) after the speed peak for ground-roll
#: deceleration onset to be recognised (corroborating/flag-independent contact
#: indicator). Migrate to config. Externalisable via the
#: ``ground_roll_decel_delta_mps`` parameter.
GROUND_ROLL_DECEL_DELTA_MPS: Final[float] = 2.5

#: Canonical Trajectory_Type labels (mirror ``TouchdownResult.trajectory_type``).
TRAJECTORY_COMPLETED_LANDING: Final[str] = "completed-landing"
TRAJECTORY_GO_AROUND: Final[str] = "go-around"
TRAJECTORY_TOUCH_AND_GO: Final[str] = "touch-and-go"
TRAJECTORY_TYPES: Final[tuple[str, ...]] = (
    TRAJECTORY_COMPLETED_LANDING,
    TRAJECTORY_GO_AROUND,
    TRAJECTORY_TOUCH_AND_GO,
)


# ---------------------------------------------------------------------------
# Result value objects (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContactSegment:
    """A maximal run of consecutive position samples in ground contact.

    ``start_time`` is the (interpolated where possible) first-contact time of the
    segment; ``min_height_m`` is the minimum height above the runway over the
    segment (``None`` when no absolute/relative height was available).
    """

    start_time: float
    end_time: float
    start_index: int
    end_index: int
    min_height_m: Optional[float]


@dataclass(frozen=True)
class TrajectoryClassification:
    """Structured trajectory classification (Req 21.1-21.5).

    ``is_touchdown`` is ``True`` only when a touchdown bracket/estimate should be
    produced: always for a completed landing, never for a go-around, and for a
    touch-and-go only under the ``"report"`` policy. ``reason_code`` carries
    :attr:`~tdz.models.FailureReason.GO_AROUND` /
    :attr:`~tdz.models.FailureReason.TOUCH_AND_GO` for the non-landing classes
    (and for a reported touch-and-go it doubles as the explicit indicator).
    """

    trajectory_type: str
    reason_code: Optional[FailureReason]
    is_touchdown: bool
    contacts: tuple[ContactSegment, ...]
    n_landings: int
    multiple_landings: bool
    first_contact_time: Optional[float]
    datum_resolved: bool
    diagnostics: dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _heights_above_runway(
    flight: FlightRecord, *, geodesy_config: object
) -> tuple[Optional[np.ndarray], bool]:
    """Return ``(heights_m, datum_resolved)`` on the position timebase.

    Heights are ``geometric_altitude - threshold_elevation_hae`` when the datum
    resolves (absolute, datum-consistent). When the datum is unresolved the
    classifier degrades to a *relative* profile (``altitude - min(altitude)``),
    flagged by ``datum_resolved=False``. ``None`` is returned only when no
    geometric altitude is available at all (rely on the on-ground flag instead).
    """
    alt = np.asarray(flight.geometric_altitudes, dtype=float)
    if alt.size == 0 or bool(np.all(np.isnan(alt))):
        return None, False

    try:
        threshold_hae = resolve_threshold_elevation_hae(flight.runway, geodesy_config)
    except DatumUnresolvedError:
        # Degrade: no absolute vertical reference -> use height above the
        # trajectory minimum so the trajectory is still classifiable.
        finite = alt[~np.isnan(alt)]
        baseline = float(np.min(finite)) if finite.size else 0.0
        return alt - baseline, False

    return alt - threshold_hae, True


def _contact_mask(
    flight: FlightRecord,
    heights: Optional[np.ndarray],
    *,
    contact_height_m: float,
) -> np.ndarray:
    """Boolean per-position-sample "in ground contact" mask.

    A sample is in contact when the on-ground flag is set OR the height above the
    runway is at/below ``contact_height_m``. The two indicators are unioned so
    contact is detected flag-independently (Req 1.2): a missing/all-False flag
    still yields contact from the altitude descent, and a missing altitude still
    yields contact from the flag.
    """
    n = int(flight.position_times.size)
    mask = np.zeros(n, dtype=bool)

    flags = np.asarray(flight.on_ground_flags, dtype=bool)
    if flags.size == n and n > 0:
        mask |= flags

    if heights is not None and heights.size == n and n > 0:
        with np.errstate(invalid="ignore"):
            mask |= np.nan_to_num(heights, nan=np.inf) <= contact_height_m

    return mask


def _interp_contact_time(
    times: np.ndarray,
    heights: Optional[np.ndarray],
    start_idx: int,
    *,
    contact_height_m: float,
) -> float:
    """Interpolate the descending crossing of ``contact_height_m`` at a segment.

    When heights are available and the sample before ``start_idx`` is airborne,
    the first-contact time is the linear crossing of ``contact_height_m`` between
    that sample and ``start_idx`` (a finer anchor than the raw sample time).
    Falls back to the segment's first sample time otherwise.
    """
    t_start = float(times[start_idx])
    if heights is None or start_idx == 0:
        return t_start
    h0 = float(heights[start_idx - 1])
    h1 = float(heights[start_idx])
    if np.isnan(h0) or np.isnan(h1) or h0 <= h1:
        return t_start
    # Descending from h0 (> contact) to h1 (<= contact): linear crossing.
    span = h0 - h1
    if span <= 0.0:
        return t_start
    frac = (h0 - contact_height_m) / span
    frac = min(max(frac, 0.0), 1.0)
    t0 = float(times[start_idx - 1])
    return t0 + frac * (t_start - t0)


def _detect_segments(
    times: np.ndarray,
    mask: np.ndarray,
    heights: Optional[np.ndarray],
    *,
    contact_height_m: float,
) -> list[ContactSegment]:
    """Group the contact mask into maximal consecutive segments."""
    segments: list[ContactSegment] = []
    n = mask.size
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if heights is not None:
            seg_h = heights[i : j + 1]
            finite = seg_h[~np.isnan(seg_h)]
            min_h: Optional[float] = float(np.min(finite)) if finite.size else None
        else:
            min_h = None
        segments.append(
            ContactSegment(
                start_time=_interp_contact_time(
                    times, heights, i, contact_height_m=contact_height_m
                ),
                end_time=float(times[j]),
                start_index=i,
                end_index=j,
                min_height_m=min_h,
            )
        )
        i = j + 1
    return segments


def _peak_decel_onset(
    velocity_times: np.ndarray,
    groundspeeds_kt: np.ndarray,
    *,
    decel_delta_mps: float,
) -> Optional[float]:
    """Ground-roll deceleration onset time (s), or ``None`` if not detected.

    The onset proxy is the time of peak groundspeed, provided the speed then
    falls by at least ``decel_delta_mps`` afterwards (a sustained decay rather
    than noise). This is a flag- and altitude-independent contact corroborator.
    """
    vt = np.asarray(velocity_times, dtype=float)
    gs = np.asarray(groundspeeds_kt, dtype=float)
    finite = ~np.isnan(gs)
    if vt.size < 2 or int(np.count_nonzero(finite)) < 2:
        return None
    gs_f = np.where(finite, gs, -np.inf)
    peak = int(np.argmax(gs_f))
    if peak >= gs.size - 1:
        return None
    after = gs[peak + 1 :]
    after = after[~np.isnan(after)]
    if after.size == 0:
        return None
    drop_mps = (float(gs[peak]) - float(np.min(after))) * KNOTS_TO_MPS
    if drop_mps < decel_delta_mps:
        return None
    return float(vt[peak])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_trajectory(
    flight: FlightRecord,
    *,
    geodesy_config: object = None,
    contact_height_m: float = CONTACT_HEIGHT_M,
    climb_out_height_m: float = CLIMB_OUT_HEIGHT_M,
    sustained_ground_roll_s: float = SUSTAINED_GROUND_ROLL_S,
    ground_roll_decel_delta_mps: float = GROUND_ROLL_DECEL_DELTA_MPS,
    touch_and_go_policy: str = "report",
) -> TrajectoryClassification:
    """Classify a flight as landing / go-around / touch-and-go (Req 21.1-21.5).

    The decision uses ground contact (flag-independent; see module docstring) and
    the post-contact vertical behaviour:

    * **no contact at all** -> go-around (``GO_AROUND``);
    * **contact, ends in sustained ground contact, no climb-out** ->
      completed-landing (bounces included);
    * **contact then a climb-out above** ``climb_out_height_m`` -> touch-and-go
      (``TOUCH_AND_GO``).

    Parameters
    ----------
    flight:
        The (QA-cleaned) flight record.
    geodesy_config:
        Optional :class:`~tdz.config.schema.GeodesyConfig`-like object forwarded
        to :func:`resolve_threshold_elevation_hae`. When the datum cannot be
        resolved the classifier degrades to a relative-altitude test
        (``datum_resolved=False``).
    contact_height_m, climb_out_height_m, sustained_ground_roll_s,
    ground_roll_decel_delta_mps:
        Externalised thresholds (documented module defaults; should migrate to
        config).
    touch_and_go_policy:
        ``"report"`` (default) tags a touch-and-go but still marks
        ``is_touchdown=True`` so its first contact can be bracketed/estimated
        with an explicit indicator; ``"suppress"`` emits a no-touchdown result
        (``is_touchdown=False``). Go-arounds never produce a touchdown regardless
        (Req 21.2, 21.3).

    Returns
    -------
    TrajectoryClassification
        The trajectory type, reason code, touchdown disposition, detected
        contact segments, multiple-landing flag and the first-contact time.
    """
    if touch_and_go_policy not in ("report", "suppress"):
        raise ValueError(
            f"touch_and_go_policy must be 'report' or 'suppress', got {touch_and_go_policy!r}"
        )

    times = np.asarray(flight.position_times, dtype=float)
    n = int(times.size)
    heights, datum_resolved = _heights_above_runway(flight, geodesy_config=geodesy_config)

    decel_onset = _peak_decel_onset(
        flight.velocity_times,
        flight.groundspeeds,
        decel_delta_mps=ground_roll_decel_delta_mps,
    )

    diagnostics: dict = {
        "datum_resolved": datum_resolved,
        "heights_available": heights is not None,
        "decel_onset_time": decel_onset,
        "n_position_samples": n,
    }

    if n == 0:
        # No position samples: cannot confirm a landing -> treat as go-around.
        return TrajectoryClassification(
            trajectory_type=TRAJECTORY_GO_AROUND,
            reason_code=FailureReason.GO_AROUND,
            is_touchdown=False,
            contacts=(),
            n_landings=0,
            multiple_landings=False,
            first_contact_time=None,
            datum_resolved=datum_resolved,
            diagnostics=diagnostics,
        )

    mask = _contact_mask(flight, heights, contact_height_m=contact_height_m)
    segments = _detect_segments(times, mask, heights, contact_height_m=contact_height_m)

    ends_in_contact = bool(mask[-1])

    # Climb-out: peak height after the FIRST contact starts. Without heights,
    # an airborne end after a contact (flag dropped) is the climb-out proxy.
    climbed_out = False
    if segments:
        first_start = segments[0].start_index
        if heights is not None:
            after = heights[first_start:]
            finite_after = after[~np.isnan(after)]
            if finite_after.size:
                climbed_out = bool(np.max(finite_after) > climb_out_height_m)
        else:
            climbed_out = not ends_in_contact

    # Multiple-landing detection: a full climb-out above climb_out_height_m
    # between two contact groups marks a separate landing (Req 21.5).
    n_landings = 1 if segments else 0
    if heights is not None and len(segments) >= 2:
        for prev, nxt in zip(segments, segments[1:]):
            between = heights[prev.end_index : nxt.start_index + 1]
            finite_between = between[~np.isnan(between)]
            if finite_between.size and float(np.max(finite_between)) > climb_out_height_m:
                n_landings += 1
    multiple_landings = n_landings > 1

    # Sustained ground roll over the final contact segment (corroboration).
    sustained = False
    if segments and ends_in_contact:
        last = segments[-1]
        sustained = (last.end_time - last.start_time) >= sustained_ground_roll_s
    diagnostics["sustained_ground_roll"] = sustained
    diagnostics["ends_in_contact"] = ends_in_contact
    diagnostics["climbed_out_after_contact"] = climbed_out

    first_contact_time = segments[0].start_time if segments else None

    # --- Decision tree --------------------------------------------------
    if not segments:
        trajectory_type = TRAJECTORY_GO_AROUND
        reason_code: Optional[FailureReason] = FailureReason.GO_AROUND
        is_touchdown = False
    elif ends_in_contact and not climbed_out:
        trajectory_type = TRAJECTORY_COMPLETED_LANDING
        reason_code = None
        is_touchdown = True
    elif climbed_out:
        trajectory_type = TRAJECTORY_TOUCH_AND_GO
        reason_code = FailureReason.TOUCH_AND_GO
        is_touchdown = touch_and_go_policy == "report"
    else:
        # Contact occurred, the record ends airborne but below the climb-out
        # height (e.g. truncated just after a low bounce). No re-takeoff
        # signature -> treat as a (short) completed landing anchored at first
        # contact rather than fabricating a go-around.
        trajectory_type = TRAJECTORY_COMPLETED_LANDING
        reason_code = None
        is_touchdown = True

    return TrajectoryClassification(
        trajectory_type=trajectory_type,
        reason_code=reason_code,
        is_touchdown=is_touchdown,
        contacts=tuple(segments),
        n_landings=n_landings,
        multiple_landings=multiple_landings,
        first_contact_time=first_contact_time,
        datum_resolved=datum_resolved,
        diagnostics=diagnostics,
    )


def classification_confusion_matrix(
    predicted: list[str],
    truth: list[str],
    *,
    labels: tuple[str, ...] = TRAJECTORY_TYPES,
) -> dict[str, dict[str, int]]:
    """Confusion matrix of predicted vs QAR-derived trajectory labels (Req 21.6).

    A small reusable helper for the classification-validation requirement. The
    full labelled-corpus validation runs in the validation harness (Task 22);
    here it supports unit testing on synthetic predicted/truth pairs and ad-hoc
    diagnostics.

    Parameters
    ----------
    predicted, truth:
        Equal-length sequences of Trajectory_Type labels.
    labels:
        The label universe (defaults to :data:`TRAJECTORY_TYPES`). Any label
        outside this set raises ``ValueError``.

    Returns
    -------
    dict
        Nested ``matrix[truth_label][predicted_label] -> count``. The diagonal
        ``matrix[L][L]`` is the count correctly classified as ``L``.
    """
    if len(predicted) != len(truth):
        raise ValueError(
            f"predicted and truth must be equal length: {len(predicted)} != {len(truth)}"
        )
    label_set = set(labels)
    matrix: dict[str, dict[str, int]] = {
        t: {p: 0 for p in labels} for t in labels
    }
    for p, t in zip(predicted, truth):
        if p not in label_set:
            raise ValueError(f"unknown predicted label: {p!r}")
        if t not in label_set:
            raise ValueError(f"unknown truth label: {t!r}")
        matrix[t][p] += 1
    return matrix
