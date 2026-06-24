"""Module 4a: Physics estimators (Task 12).

Physically interpretable touchdown anchors for the safety case -- each estimator
implements the common :class:`~tdz.models.BaseEstimator` / ``TDEstimate``
contract and inherits the Requirement-18 on-ground-flag upper bound from
:class:`PhysicsEstimator`, so no estimator can output a ``t_td`` at or after the
on-ground transition (Property 5).

Public API
----------
Shared scaffolding (Task 12.4):

* :class:`~tdz.estimators.physics.base.PhysicsEstimator` -- concrete
  :class:`~tdz.models.BaseEstimator` base; subclasses implement ``_raw_estimate``
  and set ``method_name``, and :meth:`PhysicsEstimator.estimate` applies the
  on-ground upper bound uniformly.
* :func:`~tdz.estimators.physics.base.apply_on_ground_bound` /
  :class:`~tdz.estimators.physics.base.OnGroundBoundResult` -- the bound itself
  (Req 18.1-18.4), exposed for fusion and the property test.
* :func:`~tdz.estimators.physics.base.make_estimate` /
  :func:`~tdz.estimators.physics.base.failed_estimate` -- ``TDEstimate``
  constructors that write the ``reason_code`` diagnostic consistently.
* :data:`~tdz.estimators.physics.base.CONFIDENCE_NORMAL`,
  :data:`~tdz.estimators.physics.base.CONFIDENCE_LOW`,
  :data:`~tdz.estimators.physics.base.CONFIDENCE_FAILED`,
  :data:`~tdz.estimators.physics.base.ON_GROUND_BOUND_GUARD_S` -- shared
  constants.

Deceleration-knee estimator (Task 12.1):

* :class:`~tdz.estimators.physics.decel_knee.DecelKneeEstimator` -- groundspeed
  piecewise-fit breakpoint = ``t_td`` (velocity-stream only; runs without
  geometric altitude).
* :class:`~tdz.estimators.physics.decel_knee.DecelPrior`,
  :data:`~tdz.estimators.physics.decel_knee.DEFAULT_DECEL_PRIORS`,
  :data:`~tdz.estimators.physics.decel_knee.GLOBAL_DECEL_PRIOR`,
  :func:`~tdz.estimators.physics.decel_knee.resolve_decel_prior` -- the
  aircraft-type approach-speed / rollout-deceleration priors.

Vertical flare-crossing estimator (Task 12.2):

* :class:`~tdz.estimators.physics.flare_crossing.FlareCrossingEstimator` --
  joint glideslope+flare fit over the extended region, solved for main-gear
  contact in HAE (self-disables without geometric altitude).
* :func:`~tdz.estimators.physics.flare_crossing.default_vertical_crossing_config`
  -- the design-default fit-region / bias-trigger config.

IMM filter + RTS smoother (Task 12.3):

* :class:`~tdz.estimators.physics.imm.ImmRtsEstimator` -- two-mode
  (descending vs ground roll) mode-probability crossover = ``t_td`` to
  sub-sample resolution, consuming async position/velocity natively.
"""

from tdz.estimators.physics.base import (
    CONFIDENCE_FAILED,
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
    ON_GROUND_BOUND_GUARD_S,
    OnGroundBoundResult,
    PhysicsEstimator,
    apply_on_ground_bound,
    failed_estimate,
    make_estimate,
)
from tdz.estimators.physics.decel_knee import (
    DEFAULT_DECEL_PRIORS,
    GLOBAL_DECEL_PRIOR,
    DecelKneeEstimator,
    DecelPrior,
    resolve_decel_prior,
)
from tdz.estimators.physics.flare_crossing import (
    FlareCrossingEstimator,
    default_vertical_crossing_config,
)
from tdz.estimators.physics.imm import ImmRtsEstimator

__all__ = [
    # Shared scaffolding (Task 12.4)
    "PhysicsEstimator",
    "apply_on_ground_bound",
    "OnGroundBoundResult",
    "make_estimate",
    "failed_estimate",
    "CONFIDENCE_NORMAL",
    "CONFIDENCE_LOW",
    "CONFIDENCE_FAILED",
    "ON_GROUND_BOUND_GUARD_S",
    # Deceleration-knee (Task 12.1)
    "DecelKneeEstimator",
    "DecelPrior",
    "DEFAULT_DECEL_PRIORS",
    "GLOBAL_DECEL_PRIOR",
    "resolve_decel_prior",
    # Vertical flare-crossing (Task 12.2)
    "FlareCrossingEstimator",
    "default_vertical_crossing_config",
    # IMM filter + RTS smoother (Task 12.3)
    "ImmRtsEstimator",
]
