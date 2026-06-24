"""Module 4b: Change-point estimators (Task 13).

Four independent **corroborators** of the deceleration-regime transition
(approach -> ground roll), each detecting the change on the *velocity stream* of
groundspeed with its own statistic. They run without geometric altitude (so on
every source) and emit the common :class:`~tdz.models.TDEstimate` contract,
inheriting the Requirement-18 on-ground-flag upper bound from
:class:`PhysicsEstimator` (no estimator can output a ``t_td`` at/after the
on-ground transition).

Public API
----------
Detectors (each a :class:`PhysicsEstimator` subclass; Req 5.2):

* :class:`~tdz.estimators.changepoint.pelt.PeltEstimator` (``"pelt"``) -- exact
  penalized change-point detection (piecewise-constant-mean L2 cost, BIC-like
  penalty); the change with the largest deceleration increase = ``t_td``.
* :class:`~tdz.estimators.changepoint.cusum.CusumEstimator` (``"cusum"``) --
  two-sided CUSUM for a deceleration-mean shift; the alarm is mapped back to the
  regime onset.
* :class:`~tdz.estimators.changepoint.glrt.GlrtEstimator` (``"glrt"``) --
  single-change GLR statistic for a deceleration-mean change; ``argmax`` of the
  GLR profile = ``t_td``.
* :class:`~tdz.estimators.changepoint.jerk_onset.JerkOnsetEstimator`
  (``"jerk_onset"``) -- smoothed-jerk **onset** (not peak) of the braking
  transient; CORROBORATING-only (Req 16.3): low-confidence with a
  ``corroborating_only`` diagnostics flag.

Shared scaffolding (re-exported from
:mod:`tdz.estimators.changepoint.common`):

* :class:`PhysicsEstimator`, :func:`make_estimate`, :func:`failed_estimate`,
  :data:`CONFIDENCE_NORMAL` / :data:`CONFIDENCE_LOW` / :data:`CONFIDENCE_FAILED`.
* :func:`prepare_decel_signal`, :class:`DecelSignal`,
  :class:`ChangePointSignalConfig`, :data:`DEFAULT_SIGNAL_CONFIG`,
  :func:`subsample_transition_time`, :func:`localization_sigma`.
"""

from tdz.estimators.changepoint.common import (
    CONFIDENCE_FAILED,
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
    DEFAULT_SIGNAL_CONFIG,
    ChangePointSignalConfig,
    DecelSignal,
    PhysicsEstimator,
    failed_estimate,
    localization_sigma,
    make_estimate,
    prepare_decel_signal,
    subsample_transition_time,
)
from tdz.estimators.changepoint.cusum import CusumEstimator
from tdz.estimators.changepoint.glrt import GlrtEstimator
from tdz.estimators.changepoint.jerk_onset import JerkOnsetEstimator
from tdz.estimators.changepoint.pelt import PeltEstimator

__all__ = [
    # Detectors (Task 13)
    "PeltEstimator",
    "CusumEstimator",
    "GlrtEstimator",
    "JerkOnsetEstimator",
    # Re-exported shared scaffolding
    "PhysicsEstimator",
    "make_estimate",
    "failed_estimate",
    "CONFIDENCE_NORMAL",
    "CONFIDENCE_LOW",
    "CONFIDENCE_FAILED",
    "prepare_decel_signal",
    "DecelSignal",
    "ChangePointSignalConfig",
    "DEFAULT_SIGNAL_CONFIG",
    "subsample_transition_time",
    "localization_sigma",
]
