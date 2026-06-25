"""Module 4c: Learned estimators.

LightGBM window-feature model, TCN/BiLSTM sequence model, and an optional
hybrid residual model. Trained on QAR-labeled data.

LightGBM window-feature estimator (Task 15)
-------------------------------------------
The first learned estimator (Req 5.3). Each landing is reduced to a fixed-length
engineered feature vector (:mod:`tdz.estimators.learned.features`) and fed to
gradient-boosted trees that predict a touchdown-time **offset** relative to the
physics-knee reference, with a quantile pair giving the uncertainty
(:class:`~tdz.estimators.learned.lightgbm_estimator.LightGbmTouchdownEstimator`).
It subclasses :class:`~tdz.estimators.physics.base.PhysicsEstimator`, so the
Requirement-18 on-ground-flag upper bound is inherited and applied uniformly.

Public API
----------
* :class:`~tdz.estimators.learned.lightgbm_estimator.LightGbmTouchdownEstimator`
  -- the estimator (``train``/``fit``, ``estimate``, ``feature_importances``).
* :func:`~tdz.estimators.learned.features.extract_window_features` /
  :class:`~tdz.estimators.learned.features.WindowFeatures` -- the per-landing
  feature extractor and its result type.
* :data:`~tdz.estimators.learned.features.FEATURE_NAMES`,
  :data:`~tdz.estimators.learned.features.CATEGORICAL_FEATURE_INDICES`,
  :data:`~tdz.estimators.learned.features.N_FEATURES` -- the feature schema.
"""

from tdz.estimators.learned.features import (
    CATEGORICAL_FEATURE_INDICES,
    CATEGORICAL_FEATURE_NAMES,
    DEFAULT_SIGNALS_CONFIG,
    FEATURE_NAMES,
    N_FEATURES,
    WindowFeatures,
    encode_aircraft_type,
    encode_source,
    extract_window_features,
)
from tdz.estimators.learned.lightgbm_estimator import (
    METHOD_NAME,
    LightGbmTouchdownEstimator,
)

__all__ = [
    # Estimator (Task 15)
    "LightGbmTouchdownEstimator",
    "METHOD_NAME",
    # Feature extraction
    "extract_window_features",
    "WindowFeatures",
    "FEATURE_NAMES",
    "CATEGORICAL_FEATURE_NAMES",
    "CATEGORICAL_FEATURE_INDICES",
    "N_FEATURES",
    "DEFAULT_SIGNALS_CONFIG",
    "encode_aircraft_type",
    "encode_source",
]
