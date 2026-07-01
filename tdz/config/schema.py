"""Typed, structured in-memory configuration schema.

These dataclasses mirror the design "Configuration Schema" YAML block exactly,
section for section. The top-level :class:`TDZConfig` composes one dataclass
per YAML section. Per-ICAO-type lever-arm entries are built as
:class:`tdz.config.models.LeverArm` objects and per-source descriptors as
:class:`tdz.config.models.SourceCapability` objects (reused from Task 2).

Units convention: every numeric field is SI and keeps the explicit unit suffix
from the schema (``_m``, ``_s``, ``_ft``, ``_g``, ``_deg``, ``_deg_s``, ...).
Presentation-unit suffixes (``_ft`` in vertical_crossing / fusion, etc.) are
preserved verbatim from the design schema for traceability.

This module is intentionally free of any dependency on pipeline-internal
models (:mod:`tdz.models`) to avoid import cycles; it only imports the
dependency-free config models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tdz.config.models import LeverArm, SourceCapability

__all__ = [
    "ALLOWED_ESTIMATORS",
    "ALLOWED_SOURCES",
    "ALLOWED_AIRCRAFT_CLASSES",
    "ALLOWED_GEOID_MODELS",
    "ALLOWED_ELEVATION_DATUMS",
    "ALLOWED_SPLIT_KEYS",
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
    "TDZConfig",
]


# ---------------------------------------------------------------------------
# Enumerated allowed values (module constants; validated against)
# ---------------------------------------------------------------------------

#: The complete set of estimator identifiers the system knows about. Any name
#: in ``estimators.enabled`` outside this set is rejected at startup.
ALLOWED_ESTIMATORS: frozenset[str] = frozenset(
    {
        "decel_knee",
        "flare_crossing",
        "imm_rts",
        "jerk_onset",
        "pelt",
        "cusum",
        "glrt",
        "lightgbm",
        "sequence_model",
        "hybrid_residual",
    }
)

#: ADS-B source identifiers.
ALLOWED_SOURCES: frozenset[str] = frozenset({"aireon", "fr24"})

#: Aircraft-class buckets used for lever-arm class-median defaults.
ALLOWED_AIRCRAFT_CLASSES: frozenset[str] = frozenset(
    {"regional", "narrowbody", "widebody"}
)

#: Supported geoid models for MSL -> HAE conversion.
ALLOWED_GEOID_MODELS: frozenset[str] = frozenset({"EGM2008", "EGM96", "EGM84"})

#: Datums a supplied elevation may be tagged with.
ALLOWED_ELEVATION_DATUMS: frozenset[str] = frozenset({"MSL", "HAE"})

#: Grouping keys used by the validation harness for leakage control.
ALLOWED_SPLIT_KEYS: frozenset[str] = frozenset({"tail", "airport", "runway"})


# ---------------------------------------------------------------------------
# Per-section dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    master_random_seed: int
    ads_b_source: str               # "aireon" | "fr24"


@dataclass
class TimebaseConfig:
    strategy: str                   # "common_grid" | "continuous_time"
    grid_interval_s: float
    interpolation_method: str       # "kinematic" | "linear"


@dataclass
class SignalsConfig:
    smoothing_method: str           # "savgol" | "gp"
    savgol_window_samples: int
    savgol_poly_order: int
    gp_length_scale_s: float
    gp_noise_variance: float


@dataclass
class EstimatorsConfig:
    enabled: list[str]
    physics_fallback_threshold: int  # Min QAR-labeled flights per type for learned


@dataclass
class FusionConfig:
    method: str                     # "stacking" | "weighted_blend"
    confidence_threshold_sigma: float       # Exclude/down-weight estimator if sigma_t exceeds this (seconds)
    low_confidence_ci_width_ft: float        # Flag WIDE_CONFIDENCE_INTERVAL if mapped distance 90% CI width exceeds this (feet)
    low_confidence_ci_width_s: float         # Time-domain analog: flag WIDE_CONFIDENCE_INTERVAL if fused 90% CI width exceeds this (seconds)
    disagreement_threshold_s: float          # Flag ESTIMATOR_DISAGREEMENT if inter-estimator spread (1-sigma) exceeds this (seconds)


@dataclass
class QualityGatesConfig:
    min_samples_near_td: int
    max_gap_spanning_td_s: float
    min_samples_in_window: int
    window_half_width_s: float
    max_excluded_fraction: float
    max_longitudinal_accel_g: float
    max_lateral_accel_g: float
    max_turn_rate_deg_s: float
    duplicate_timestamp_tolerance_s: float


@dataclass
class ClassMedian:
    """Median lever-arm values for an aircraft class (missing-type fallback)."""

    vertical_offset_m: float
    longitudinal_offset_m: float
    nominal_touchdown_pitch_deg: float


@dataclass
class LeverArmsConfig:
    """Lever-arm table plus class-median defaults and CI policy."""

    arms: dict[str, LeverArm]                       # Per ICAO type designator
    default_strategy: str                           # "class_median"
    class_medians: dict[str, ClassMedian]           # Keyed by aircraft class
    class_default_widens_ci: bool


@dataclass
class GeodesyConfig:
    geoid_model: str                        # "EGM2008" | ...
    assume_runway_elevation_datum: str      # "MSL" | "HAE"


@dataclass
class VerticalCrossingConfig:
    fit_region_upper_ft: float
    fit_region_lower_ft: float
    min_samples_in_fit_region: int
    residual_bias_trigger_ft: float


@dataclass
class ValidationConfig:
    primary_split_key: str                  # "tail" | "airport" | "runway"
    generalization_evals: list[str]
    use_calibration_split: bool
    min_stratum_size: int
    cross_source: bool
    clock_offset_max_s: float
    clock_drift_max_s: float
    wrong_runway_lateral_margin_ft: float


@dataclass
class OutputConfig:
    distance_units: str                     # "feet" | "meters"
    speed_units: str                        # "knots" | "mps"
    time_precision_decimals: int


# ---------------------------------------------------------------------------
# Top-level composed config
# ---------------------------------------------------------------------------


@dataclass
class TDZConfig:
    """Fully-resolved, validated configuration object tree.

    ``resolved`` holds the complete resolved configuration as a plain dict
    (including any applied defaults) for reproducibility / provenance hashing
    (Req 20.5). Use :meth:`to_dict` / :meth:`to_yaml` to export it.
    """

    pipeline: PipelineConfig
    timebase: TimebaseConfig
    signals: SignalsConfig
    estimators: EstimatorsConfig
    fusion: FusionConfig
    quality_gates: QualityGatesConfig
    lever_arms: LeverArmsConfig
    geodesy: GeodesyConfig
    vertical_crossing: VerticalCrossingConfig
    sources: dict[str, SourceCapability]
    validation: ValidationConfig
    output: OutputConfig

    # Resolved configuration (defaults applied) as a plain dict, for provenance.
    resolved: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return the fully-resolved configuration as a plain dict."""
        return self.resolved

    def to_yaml(self) -> str:
        """Re-serialize the fully-resolved configuration to YAML."""
        import yaml

        return yaml.safe_dump(self.resolved, sort_keys=True, default_flow_style=False)
