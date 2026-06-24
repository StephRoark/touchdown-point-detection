"""Module 1b: Trajectory classification & coarse bracket.

Classify a trajectory as completed-landing, go-around, or touch-and-go, and form
a flag-independent coarse touchdown bracket (first pass) used by all downstream
quality gates. Classification runs *before* any estimator (design "Trajectory
classified before estimating"): only a completed landing (or a reported
touch-and-go) yields a touchdown bracket; a go-around short-circuits to a
no-touchdown result with no window. The bracket is anchored to the **first**
main-gear contact so a bounce never averages across two contacts (Req 21.4 /
Property 21).

Public API (Task 10.1 -- trajectory classification, :mod:`tdz.bracket.classify`):

* :func:`classify_trajectory` -- classify a :class:`~tdz.models.FlightRecord`
  as completed-landing / go-around / touch-and-go using flag-independent ground
  contact and the post-contact vertical profile (Req 21.1-21.5).
* :class:`TrajectoryClassification` -- the frozen result (trajectory type,
  reason code, touchdown disposition, contact segments, multiple-landing flag,
  first-contact time, datum-resolved flag, diagnostics).
* :class:`ContactSegment` -- a maximal run of consecutive in-contact samples.
* :func:`classification_confusion_matrix` -- predicted-vs-truth confusion matrix
  over the trajectory-type labels (Req 21.6).
* :data:`TRAJECTORY_TYPES` and the label constants
  :data:`TRAJECTORY_COMPLETED_LANDING` / :data:`TRAJECTORY_GO_AROUND` /
  :data:`TRAJECTORY_TOUCH_AND_GO`.
* Threshold constants :data:`CONTACT_HEIGHT_M`, :data:`CLIMB_OUT_HEIGHT_M`,
  :data:`SUSTAINED_GROUND_ROLL_S`, :data:`GROUND_ROLL_DECEL_DELTA_MPS`
  (documented defaults; externalisable per call; SHOULD migrate to config).

Public API (Task 10.2 -- coarse bracket, :mod:`tdz.bracket.coarse_bracket`):

* :func:`compute_coarse_bracket` -- form the first-pass coarse touchdown bracket
  ``[t_lo, t_hi]`` from the on-ground flag (upper bound only) and the
  flag-independent altitude-descent + deceleration-onset indicators, anchored to
  the first contact (Req 1.1, 1.2; Req 18.2). It reuses
  :class:`~tdz.io.qa.TouchdownWindow` as the window type so
  ``run_qa(..., touchdown_window=bracket.window)`` consumes the real bracket.
* :class:`BracketResult` -- the frozen outcome (``"ok"`` with a window, or
  ``"no-touchdown"`` with a reason code and no window).
* :data:`DEFAULT_BRACKET_HALF_WIDTH_S` -- default bracket half-width (callers
  SHOULD pass ``QualityGatesConfig.window_half_width_s``).
* Indicator id constants :data:`INDICATOR_ON_GROUND_FLAG`,
  :data:`INDICATOR_ALTITUDE_DESCENT`, :data:`INDICATOR_DECEL_ONSET`.
"""

from tdz.bracket.classify import (
    CLIMB_OUT_HEIGHT_M,
    CONTACT_HEIGHT_M,
    GROUND_ROLL_DECEL_DELTA_MPS,
    SUSTAINED_GROUND_ROLL_S,
    TRAJECTORY_COMPLETED_LANDING,
    TRAJECTORY_GO_AROUND,
    TRAJECTORY_TOUCH_AND_GO,
    TRAJECTORY_TYPES,
    ContactSegment,
    TrajectoryClassification,
    classification_confusion_matrix,
    classify_trajectory,
)
from tdz.bracket.coarse_bracket import (
    DEFAULT_BRACKET_HALF_WIDTH_S,
    INDICATOR_ALTITUDE_DESCENT,
    INDICATOR_DECEL_ONSET,
    INDICATOR_ON_GROUND_FLAG,
    BracketResult,
    compute_coarse_bracket,
)

__all__ = [
    # classification (Task 10.1)
    "classify_trajectory",
    "TrajectoryClassification",
    "ContactSegment",
    "classification_confusion_matrix",
    "TRAJECTORY_TYPES",
    "TRAJECTORY_COMPLETED_LANDING",
    "TRAJECTORY_GO_AROUND",
    "TRAJECTORY_TOUCH_AND_GO",
    "CONTACT_HEIGHT_M",
    "CLIMB_OUT_HEIGHT_M",
    "SUSTAINED_GROUND_ROLL_S",
    "GROUND_ROLL_DECEL_DELTA_MPS",
    # coarse bracket (Task 10.2)
    "compute_coarse_bracket",
    "BracketResult",
    "DEFAULT_BRACKET_HALF_WIDTH_S",
    "INDICATOR_ON_GROUND_FLAG",
    "INDICATOR_ALTITUDE_DESCENT",
    "INDICATOR_DECEL_ONSET",
]
