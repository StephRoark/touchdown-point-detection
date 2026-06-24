"""Module 3: Signals / feature construction (Task 11).

Segmented regression on raw groundspeed is the **primary** deceleration-regime
estimate (the breakpoint is the regime transition); non-stationary / piecewise
Savitzky-Golay or GP-surrogate derivatives provide **corroborating** signals
only; and feature channels (incl. distance-to-threshold and the time-delta
channel) feed the learned estimators.

Public API
----------
Segmented regression (Task 11.1):

* :func:`~tdz.signals.segmented.fit_segmented_groundspeed` -- fit a continuous
  piecewise-linear model to raw groundspeed; the breakpoint is the
  deceleration-regime transition (Req 16.1).
* :class:`~tdz.signals.segmented.SegmentedFit` -- frozen result (breakpoint
  time, per-segment slopes in m/s^2, intercepts, residual RMS).

Corroborating derivatives (Task 11.2):

* :func:`~tdz.signals.derivatives.smoothed_derivatives` -- deceleration and jerk
  via SavGol or a non-stationary/piecewise GP surrogate, with per-sample
  posterior std and a reliability flag (Req 16.2-16.6).
* :class:`~tdz.signals.derivatives.DerivativeResult` -- frozen result.
* :func:`~tdz.signals.derivatives.deceleration_rms_discrepancy` -- QAR-vs-smoothed
  RMS-discrepancy harness (Req 16.7).

Feature channels (Task 11.3):

* :func:`~tdz.signals.features.build_feature_channels` -- build all learned-model
  feature channels (separate position/velocity timebases).
* :func:`~tdz.signals.features.populate_flight_record` -- populate the
  :class:`~tdz.models.FlightRecord` derived-signal slots in place.
* :class:`~tdz.signals.features.FeatureChannels` -- frozen channel bundle.
"""

from tdz.signals.derivatives import (
    GP_KERNEL_SUPPORT_SIGMAS,
    GP_POLY_ORDER,
    MIN_VALID_SAMPLES_IN_WINDOW,
    DerivativeResult,
    deceleration_rms_discrepancy,
    smoothed_derivatives,
)
from tdz.signals.features import (
    FeatureChannels,
    build_feature_channels,
    populate_flight_record,
)
from tdz.signals.segmented import (
    MIN_SAMPLES_PER_SEGMENT,
    SegmentedFit,
    fit_segmented_groundspeed,
)

__all__ = [
    # Segmented regression (primary decel-regime estimate)
    "SegmentedFit",
    "fit_segmented_groundspeed",
    "MIN_SAMPLES_PER_SEGMENT",
    # Corroborating smoothed derivatives
    "DerivativeResult",
    "smoothed_derivatives",
    "deceleration_rms_discrepancy",
    "MIN_VALID_SAMPLES_IN_WINDOW",
    "GP_KERNEL_SUPPORT_SIGMAS",
    "GP_POLY_ORDER",
    # Feature channels
    "FeatureChannels",
    "build_feature_channels",
    "populate_flight_record",
]
