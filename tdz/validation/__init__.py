"""Module 7: Validation harness.

Tail-grouped split + held-out-airport/runway evaluations + calibration split,
stratified metrics, clock-independent distance truth, and cross-source
evaluation.

Clock alignment (Task 21) is the first component: it estimates the per-flight
QAR<->ADS-B clock offset by cross-correlating an overlapping kinematic series,
detects within-flight drift, reports the corpus offset distribution, and
excludes over-threshold/unreliable flights from the time domain while retaining
them for clock-independent distance validation (Requirement 19).
"""

from tdz.validation.cross_source import (    CrossSourceArm,
    CrossSourceDirection,
    CrossSourceReport,
    default_landing_key,
    evaluate_cross_source,
    shared_landings,
)
from tdz.validation.clock_alignment import (
    QUALITY_DEGRADED,
    QUALITY_FAILED,
    QUALITY_GOOD,
    ClockAlignmentReport,
    ClockOffsetResult,
    KinematicSeries,
    OffsetDistribution,
    align_corpus,
    apply_offset_to_qar,
    estimate_clock_offset,
)
from tdz.validation.metrics import (
    NO_ESTIMATE,
    FlightEvaluation,
    StratifiedMetricsReport,
    StratumResult,
    along_runway_truth_distance_ft,
    approach_speed_band_label,
    compute_metrics,
    compute_stratified_metrics,
)
from tdz.validation.coverage import (
    COVERAGE_IN_BAND,
    COVERAGE_OVER,
    COVERAGE_UNDEFINED,
    COVERAGE_UNDER,
    KNOTS_TO_FT_PER_S,
    BelowTargetFlag,
    CoverageAssessment,
    ErrorFloorReport,
    assess_coverage,
    cadence_limited_floor_ft,
    characterize_error_floor,
    classify_coverage,
    flag_below_target,
)
from tdz.validation.splits import (
    FlightGroupKeys,
    GeneralizationSplit,
    GroupedSplit,
    ValidationSplits,
    group_keys_from_records,
    make_generalization_split,
    make_primary_split,
    make_validation_splits,
)

__all__ = [
    "QUALITY_GOOD",
    "QUALITY_DEGRADED",
    "QUALITY_FAILED",
    "KinematicSeries",
    "ClockOffsetResult",
    "OffsetDistribution",
    "ClockAlignmentReport",
    "estimate_clock_offset",
    "align_corpus",
    "apply_offset_to_qar",
    "FlightGroupKeys",
    "GroupedSplit",
    "GeneralizationSplit",
    "ValidationSplits",
    "group_keys_from_records",
    "make_primary_split",
    "make_generalization_split",
    "make_validation_splits",
    "NO_ESTIMATE",
    "FlightEvaluation",
    "StratumResult",
    "StratifiedMetricsReport",
    "along_runway_truth_distance_ft",
    "approach_speed_band_label",
    "compute_metrics",
    "compute_stratified_metrics",
    "COVERAGE_IN_BAND",
    "COVERAGE_OVER",
    "COVERAGE_UNDER",
    "COVERAGE_UNDEFINED",
    "KNOTS_TO_FT_PER_S",
    "CoverageAssessment",
    "ErrorFloorReport",
    "BelowTargetFlag",
    "classify_coverage",
    "assess_coverage",
    "cadence_limited_floor_ft",
    "characterize_error_floor",
    "flag_below_target",
    "CrossSourceArm",
    "CrossSourceDirection",
    "CrossSourceReport",
    "default_landing_key",
    "shared_landings",
    "evaluate_cross_source",
]
