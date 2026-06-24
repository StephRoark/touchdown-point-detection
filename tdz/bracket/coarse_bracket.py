"""Flag-independent coarse touchdown bracket (first pass) -- Task 10.2.

The detector runs in two passes (design "Two-pass: coarse bracket -> sub-sample
estimate", Req 1.1): this module produces the **first-pass coarse bracket** --
a bounded time window ``[t_lo, t_hi]`` expected to contain ``t_td`` -- against
which all downstream data-sufficiency / quality gates are evaluated (the final
``t_td`` is not yet available when gating occurs). It closes the chicken-and-egg
the QA module documented: the bracket is emitted as the very same
:class:`~tdz.io.qa.TouchdownWindow` the sufficiency gate consumes (reused, not
duplicated), so ``run_qa(..., touchdown_window=bracket.window)`` references the
real bracket.

How the bracket is formed
-------------------------
For a completed landing (or a reported touch-and-go) the bracket is built from
**both** indicators (Req 1.2):

1. **On-ground flag transition as an UPPER bound only** (design "On-ground flag
   as upper bracket only", Req 18.2). The flag transitions with a delay *after*
   real touchdown, so ``t_td <= on_ground_transition_time``; it is used as
   ``t_hi`` (and to clamp the first-contact anchor) but is **never** the answer.
2. **A flag-independent indicator** -- the descent of geometric altitude toward
   the (HAE-resolved) runway elevation **together with** the onset of ground-roll
   deceleration. This lets a bracket form when the on-ground flag is absent,
   delayed or missing, and even when the vertical datum is unresolved (then the
   deceleration onset / on-ground flag carry the anchor; see ``datum_resolved``).

First-contact anchoring (bounce handling, Req 21.4 / Property 21)
-----------------------------------------------------------------
The bracket is anchored to the **first** main-gear contact (the start of the
first contact segment from :func:`classify_trajectory`), never to a value
averaged across a bounce. When multiple contacts exist the bracket targets the
first, so the second-pass estimators do not return a physically meaningless
midpoint.

Degradation
-----------
* **Go-around** -> no-touchdown :class:`BracketResult` with
  :attr:`~tdz.models.FailureReason.GO_AROUND`, **no window** (Req 21.2).
* **Touch-and-go** -> tagged; a window is produced under the ``"report"`` policy
  (default) or suppressed (``TOUCH_AND_GO`` no-touchdown) under ``"suppress"``.
* **Datum unresolved** -> the absolute altitude-descent indicator is unavailable;
  the bracket degrades to the deceleration-onset and on-ground-flag anchors
  (``datum_resolved=False``, surfaced in diagnostics). If *no* anchor can be
  formed at all, a no-touchdown result is returned.

Half-width
----------
``half_width_s`` defaults to :data:`DEFAULT_BRACKET_HALF_WIDTH_S` but callers
SHOULD pass ``QualityGatesConfig.window_half_width_s`` so the bracket and the
sufficiency gate share one externalised half-width.

Units convention
----------------
SI internally: times in epoch seconds, heights in metres (HAE). No conversion to
feet/knots happens here (that is the output boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from tdz.bracket.classify import (
    CLIMB_OUT_HEIGHT_M,
    CONTACT_HEIGHT_M,
    GROUND_ROLL_DECEL_DELTA_MPS,
    SUSTAINED_GROUND_ROLL_S,
    TrajectoryClassification,
    classify_trajectory,
)
from tdz.io.qa import TouchdownWindow
from tdz.models import FailureReason, FlightRecord

__all__ = [
    "DEFAULT_BRACKET_HALF_WIDTH_S",
    "INDICATOR_ON_GROUND_FLAG",
    "INDICATOR_ALTITUDE_DESCENT",
    "INDICATOR_DECEL_ONSET",
    "BracketResult",
    "compute_coarse_bracket",
]

#: Default coarse-bracket half-width (s). Callers SHOULD pass
#: ``QualityGatesConfig.window_half_width_s`` instead so the bracket and the QA
#: sufficiency gate share one externalised value (the QA default is also 30 s).
DEFAULT_BRACKET_HALF_WIDTH_S: Final[float] = 30.0

#: Stable identifiers for which indicators contributed to the bracket.
INDICATOR_ON_GROUND_FLAG: Final[str] = "on_ground_flag"
INDICATOR_ALTITUDE_DESCENT: Final[str] = "altitude_descent"
INDICATOR_DECEL_ONSET: Final[str] = "decel_onset"


@dataclass(frozen=True)
class BracketResult:
    """Coarse-bracket outcome for one flight (first pass).

    ``status`` is ``"ok"`` (a touchdown bracket was formed) or ``"no-touchdown"``
    (go-around, suppressed touch-and-go, or no usable anchor). On ``"ok"``,
    ``window`` is the :class:`~tdz.io.qa.TouchdownWindow` downstream gates
    consume, with ``center`` at the first-contact anchor; ``window`` is ``None``
    otherwise. ``indicators_fired`` records which of
    :data:`INDICATOR_ON_GROUND_FLAG` / :data:`INDICATOR_ALTITUDE_DESCENT` /
    :data:`INDICATOR_DECEL_ONSET` contributed.
    """

    status: str
    window: Optional[TouchdownWindow]
    first_contact_time: Optional[float]
    trajectory_type: str
    reason_code: Optional[FailureReason]
    indicators_fired: tuple[str, ...]
    on_ground_upper_bound: Optional[float]
    datum_resolved: bool
    classification: TrajectoryClassification
    diagnostics: dict


def compute_coarse_bracket(
    flight: FlightRecord,
    *,
    geodesy_config: object = None,
    half_width_s: float = DEFAULT_BRACKET_HALF_WIDTH_S,
    classification: Optional[TrajectoryClassification] = None,
    contact_height_m: float = CONTACT_HEIGHT_M,
    climb_out_height_m: float = CLIMB_OUT_HEIGHT_M,
    sustained_ground_roll_s: float = SUSTAINED_GROUND_ROLL_S,
    ground_roll_decel_delta_mps: float = GROUND_ROLL_DECEL_DELTA_MPS,
    touch_and_go_policy: str = "report",
) -> BracketResult:
    """Form the first-pass coarse touchdown bracket (Req 1.1, 1.2; Module 1b).

    Classifies the trajectory (unless ``classification`` is supplied) and, for a
    completed landing or reported touch-and-go, builds ``[t_lo, t_hi]`` from the
    on-ground flag (upper bound only) and the flag-independent altitude-descent +
    deceleration-onset indicators, anchored to the first contact (see module
    docstring). Go-arounds and suppressed touch-and-goes return a no-touchdown
    :class:`BracketResult` with the matching reason code and no window.

    Parameters
    ----------
    flight:
        The (QA-cleaned) flight record.
    geodesy_config:
        Optional :class:`~tdz.config.schema.GeodesyConfig`-like object forwarded
        to datum resolution; an unresolved datum degrades gracefully.
    half_width_s:
        Bracket half-width (s); pass ``QualityGatesConfig.window_half_width_s``.
    classification:
        A precomputed :class:`TrajectoryClassification` to reuse; classified
        afresh when ``None`` (using the threshold/policy parameters below).
    contact_height_m, climb_out_height_m, sustained_ground_roll_s,
    ground_roll_decel_delta_mps, touch_and_go_policy:
        Forwarded to :func:`classify_trajectory` when classifying here.

    Returns
    -------
    BracketResult
        ``status`` ``"ok"`` with a :class:`~tdz.io.qa.TouchdownWindow`, or
        ``"no-touchdown"`` with a reason code and no window.
    """
    if classification is None:
        classification = classify_trajectory(
            flight,
            geodesy_config=geodesy_config,
            contact_height_m=contact_height_m,
            climb_out_height_m=climb_out_height_m,
            sustained_ground_roll_s=sustained_ground_roll_s,
            ground_roll_decel_delta_mps=ground_roll_decel_delta_mps,
            touch_and_go_policy=touch_and_go_policy,
        )

    on_ground_upper = flight.on_ground_transition_time
    decel_onset = classification.diagnostics.get("decel_onset_time")
    datum_resolved = classification.datum_resolved

    base_diag = {
        "n_landings": classification.n_landings,
        "multiple_landings": classification.multiple_landings,
        "decel_onset_time": decel_onset,
        "datum_resolved": datum_resolved,
    }

    # Non-touchdown trajectories short-circuit with no window (Req 21.2/21.3).
    if not classification.is_touchdown:
        return BracketResult(
            status="no-touchdown",
            window=None,
            first_contact_time=classification.first_contact_time,
            trajectory_type=classification.trajectory_type,
            reason_code=classification.reason_code,
            indicators_fired=(),
            on_ground_upper_bound=on_ground_upper,
            datum_resolved=datum_resolved,
            classification=classification,
            diagnostics=base_diag,
        )

    # --- Assemble the first-contact anchor from the available indicators ---
    indicators: list[str] = []

    # Altitude-descent anchor: the first contact segment's (interpolated)
    # first-contact time, available only when an absolute height descent could
    # be measured (datum resolved + geometric altitude present).
    altitude_anchor: Optional[float] = None
    if (
        datum_resolved
        and classification.diagnostics.get("heights_available")
        and classification.contacts
    ):
        altitude_anchor = classification.contacts[0].start_time
        indicators.append(INDICATOR_ALTITUDE_DESCENT)

    if decel_onset is not None:
        indicators.append(INDICATOR_DECEL_ONSET)

    if on_ground_upper is not None:
        indicators.append(INDICATOR_ON_GROUND_FLAG)

    # Primary anchor preference: altitude descent -> deceleration onset ->
    # first contact segment start (flag-driven) -> on-ground transition.
    if altitude_anchor is not None:
        first_contact_time: Optional[float] = altitude_anchor
    elif decel_onset is not None:
        first_contact_time = float(decel_onset)
    elif classification.first_contact_time is not None:
        first_contact_time = classification.first_contact_time
    else:
        first_contact_time = on_ground_upper

    if first_contact_time is None:
        # No usable anchor (no altitude, no deceleration, no flag): cannot form
        # a bracket -> degrade to no-touchdown rather than fabricate a window.
        return BracketResult(
            status="no-touchdown",
            window=None,
            first_contact_time=None,
            trajectory_type=classification.trajectory_type,
            reason_code=FailureReason.INSUFFICIENT_SAMPLES,
            indicators_fired=tuple(indicators),
            on_ground_upper_bound=on_ground_upper,
            datum_resolved=datum_resolved,
            classification=classification,
            diagnostics=base_diag,
        )

    # On-ground flag is an UPPER bound only: clamp the anchor to it and use it as
    # t_hi (Req 18.2). Otherwise the upper bound is anchor + half_width.
    if on_ground_upper is not None:
        first_contact_time = min(first_contact_time, float(on_ground_upper))
        t_hi = float(on_ground_upper)
    else:
        t_hi = first_contact_time + half_width_s

    t_lo = first_contact_time - half_width_s
    if t_hi <= t_lo:
        # Degenerate (e.g. flag transition at/under the lower bound): restore a
        # symmetric window around the anchor so t_lo < t_hi always holds.
        t_hi = first_contact_time + half_width_s

    window = TouchdownWindow(t_lo=t_lo, t_hi=t_hi, center=first_contact_time)

    return BracketResult(
        status="ok",
        window=window,
        first_contact_time=first_contact_time,
        trajectory_type=classification.trajectory_type,
        reason_code=classification.reason_code,
        indicators_fired=tuple(indicators),
        on_ground_upper_bound=on_ground_upper,
        datum_resolved=datum_resolved,
        classification=classification,
        diagnostics=base_diag,
    )
