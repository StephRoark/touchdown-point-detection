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
    "UncertaintyConfig",
    "QualityGatesConfig",
    "ClassMedian",
    "LeverArmsConfig",
    "GeodesyConfig",
    "VerticalCrossingConfig",
    "ValidationConfig",
    "ProvisionalAccuracyTargets",
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
    # Explicit neural deterministic-execution mode (Req 15.2). When True the
    # neural sequence model requests PyTorch deterministic algorithms so its
    # outputs are bit-identical across runs (at reduced throughput); when False
    # the neural model is only guaranteed reproducible within a documented
    # tolerance. Physics / change-point / LightGBM / geometry are always
    # bit-identical regardless of this flag. The mode used is recorded in the
    # batch provenance. Defaulted so existing constructors stay valid.
    deterministic_mode: bool = True


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
class UncertaintyConfig:
    """Uncertainty quantification / calibration knobs (Task 19).

    Governs conformal calibration (``coverage_target``), gap-proportional
    interval widening (``nominal_cadence_s``, ``gap_window_half_width_s``,
    ``gap_min_duration_s``; Req 9.2), missing-lever-arm distance-CI widening
    (``missing_lever_arm_widening_factor``; Req 7.5), and post-transition
    sample-starvation widening (``post_transition_window_s``,
    ``min_post_transition_samples``, ``starvation_widening_factor``; Req 9.6).
    """

    coverage_target: float                  # Target empirical coverage for reported CIs (e.g. 0.90)
    nominal_cadence_s: float                # Nominal ADS-B sample interval C (seconds) for gap widening
    gap_window_half_width_s: float          # Consider gaps within ±this window of t_td (seconds)
    gap_min_duration_s: float               # Only gaps exceeding this duration widen the interval (seconds)
    missing_lever_arm_widening_factor: float  # Distance-CI widening factor when a class-median lever arm is used
    post_transition_window_s: float         # Window after on-ground transition to look for ground-roll samples (seconds)
    min_post_transition_samples: int        # Fewer than this many post-transition samples -> starvation widening
    starvation_widening_factor: float       # CI widening factor applied on post-transition sample starvation


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
class ProvisionalAccuracyTargets:
    """Provisional accuracy targets used for below-target flagging (Req 13).

    **PROVISIONAL.** These are *reporting* targets, not pass/fail gates: until
    they are ratified against the empirically characterized cadence-limited
    error floor (Req 13.0), the harness reports observed metrics against them
    and *flags* below-target strata rather than failing (Req 13.5). Every value
    is in feet or percent, matching :class:`~tdz.models.ValidationMetrics`.
    """

    distance_rmse_ft: float                 # Overall RMSE target (Req 13.1; <=250 ft provisional)
    distance_p95_abs_error_ft: float        # 95th-pct absolute error target (Req 13.3; <=400 ft)
    distance_p95_long_side_ft: float        # 95th-pct positive/long-side error cap (Req 13.3; <=500 ft)
    median_signed_error_abs_ft: float       # |median signed error| bias cap (Req 13.2; <=75 ft)
    baseline_improvement_pct: float         # Min RMSE reduction vs naive baseline (Req 13.4; >=30%)


@dataclass
class ValidationConfig:
    primary_split_key: str                  # "tail" | "airport" | "runway"
    generalization_evals: list[str]
    use_calibration_split: bool
    # Primary three-way grouped-split fractions (train / calibration / test).
    # Whole groups (under primary_split_key) are assigned to exactly one part;
    # fractions are of the *group* population and are normalized by their sum
    # (Req 12.2, 12.4). Driven by the master random seed for reproducibility.
    train_fraction: float
    calibration_fraction: float
    test_fraction: float
    min_stratum_size: int
    cross_source: bool
    clock_offset_max_s: float
    clock_drift_max_s: float
    clock_xcorr_resample_dt_s: float
    clock_max_lag_search_s: float
    clock_min_overlap_s: float
    clock_min_peak_correlation: float
    clock_drift_segments: int
    wrong_runway_lateral_margin_ft: float
    # Approach-speed-band edges (knots) used to bucket flights for stratified
    # metric reporting (Req 12.7). Strictly increasing; N edges define N+1 bands
    # ("<e0", "e0-e1", ..., ">=e_{N-1}"). Reporting-only -- these never enter
    # estimation numerics. Defaulted so existing constructors remain valid.
    approach_speed_band_edges_kt: tuple[float, ...] = (120.0, 140.0, 160.0)
    # Empirical-coverage acceptance band for the reported 90% CIs (Req 4.3, 4.4).
    # Coverage inside [coverage_min, coverage_max] is acceptable; below
    # coverage_min is undercovered (unsafe); above coverage_max is overcovered
    # (uninformative). Reporting-only -- never enters estimation numerics.
    coverage_min: float = 0.85
    coverage_max: float = 0.95
    # Minimum stratum size for below-target flagging (Req 13.5). Distinct from
    # min_stratum_size (the >=30 reporting gate in 22.2): a stratum is only
    # flagged below-target when it holds at least this many flights.
    below_target_min_flights: int = 200
    # Provisional accuracy targets (Req 13) used for below-target flagging.
    provisional_targets: "ProvisionalAccuracyTargets" = field(
        default_factory=lambda: ProvisionalAccuracyTargets(
            distance_rmse_ft=250.0,
            distance_p95_abs_error_ft=400.0,
            distance_p95_long_side_ft=500.0,
            median_signed_error_abs_ft=75.0,
            baseline_improvement_pct=30.0,
        )
    )


@dataclass
class OutputConfig:
    """Output-boundary presentation knobs (Task 20).

    This is the single SI->presentation conversion point in the pipeline. The
    speed plausibility band (``speed_min_kt`` / ``speed_max_kt``), the reported
    speed resolution (``speed_resolution_kt``), and the window within which
    velocity samples must exist for a confident touchdown speed
    (``speed_velocity_gap_max_s``) are all tunables governing Req 3.1 / 3.4.
    """

    distance_units: str                     # "feet" | "meters"
    speed_units: str                        # "knots" | "mps"
    time_precision_decimals: int
    speed_min_kt: float                     # Lower bound of the plausible touchdown-speed band (knots; Req 3.1)
    speed_max_kt: float                     # Upper bound of the plausible touchdown-speed band (knots; Req 3.1)
    speed_resolution_kt: float              # Reported groundspeed resolution (knots; Req 3.1)
    speed_velocity_gap_max_s: float         # Flag speed low-confidence if no velocity sample within this window of t_td (seconds; Req 3.4)


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
    uncertainty: UncertaintyConfig
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
