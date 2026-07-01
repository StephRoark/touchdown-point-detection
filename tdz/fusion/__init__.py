"""Module 5: Fusion / ensemble.

Calibrated weighted blend / stacking with reliability weighting that combines
estimator outputs into a single fused estimate with a predictive interval.
"""

from tdz.fusion.ensemble import (
    CI_COVERAGE,
    CONFIDENCE_NO_ESTIMATE,
    METHOD_STACKING,
    METHOD_WEIGHTED_BLEND,
    ON_GROUND_FLAG_METHOD_NAMES,
    CalibratedFusion,
    build_fusion,
)

__all__ = [
    "CI_COVERAGE",
    "CONFIDENCE_NO_ESTIMATE",
    "METHOD_STACKING",
    "METHOD_WEIGHTED_BLEND",
    "ON_GROUND_FLAG_METHOD_NAMES",
    "CalibratedFusion",
    "build_fusion",
]
