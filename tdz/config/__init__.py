"""Configuration.

Lever-arm tables, geoid model selection, source descriptors, thresholds,
method selection, schema validation, and defaults resolution.

Public surface:
- :class:`LeverArm`, :class:`SourceCapability` — dependency-free config models
  (Task 2), reused when building the typed config tree.
- :class:`TDZConfig` and the per-section dataclasses — the typed schema.
- :func:`load_config` / :func:`build_config` — load + validate + resolve.
- :class:`ConfigValidationError` — fail-fast validation error.
"""

from tdz.config.errors import ConfigError, ConfigValidationError
from tdz.config.loader import build_config, load_config
from tdz.config.models import LeverArm, SourceCapability
from tdz.config.schema import (
    ALLOWED_ESTIMATORS,
    ClassMedian,
    EstimatorsConfig,
    FusionConfig,
    GeodesyConfig,
    LeverArmsConfig,
    OutputConfig,
    PipelineConfig,
    QualityGatesConfig,
    SignalsConfig,
    TDZConfig,
    TimebaseConfig,
    ValidationConfig,
    VerticalCrossingConfig,
)

__all__ = [
    "LeverArm",
    "SourceCapability",
    "TDZConfig",
    "PipelineConfig",
    "TimebaseConfig",
    "SignalsConfig",
    "EstimatorsConfig",
    "FusionConfig",
    "QualityGatesConfig",
    "ClassMedian",
    "LeverArmsConfig",
    "GeodesyConfig",
    "VerticalCrossingConfig",
    "ValidationConfig",
    "OutputConfig",
    "ALLOWED_ESTIMATORS",
    "load_config",
    "build_config",
    "ConfigError",
    "ConfigValidationError",
]
