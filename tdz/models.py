"""Pipeline-internal and boundary data models.

This module defines the shared dataclasses and abstract interfaces that flow
through the touchdown-detection pipeline, plus the input-schema records and the
:class:`FailureReason` enumeration.

Units convention
----------------
All internal/pipeline-internal fields are held in SI units (meters,
meters/second, seconds, radians) unless the field name says otherwise. The only
fields expressed in presentation units (feet, knots) live on
:class:`TouchdownResult`, which is the output boundary; those fields carry an
explicit ``_ft`` / ``_kt`` suffix.

Abstract interfaces (:class:`BaseEstimator`, :class:`FusionEnsemble`) declare
the method contracts only; their methods raise :class:`NotImplementedError` and
are implemented by later tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from tdz.config.models import LeverArm, SourceCapability

__all__ = [
    "TDEstimate",
    "BaseEstimator",
    "RunwayReference",
    "FlightRecord",
    "FusedEstimate",
    "FusionEnsemble",
    "TouchdownResult",
    "AireonMessage",
    "FR24Record",
    "SourceCapability",
    "LeverArm",
    "QARTruthRecord",
    "ValidationMetrics",
    "FailureReason",
]


# ---------------------------------------------------------------------------
# Common estimator interface
# ---------------------------------------------------------------------------


@dataclass
class TDEstimate:
    """Common output contract for all touchdown estimators."""

    t_td: float                     # Estimated touchdown time (epoch seconds, sub-sample)
    sigma_t: float                  # 1-sigma uncertainty in seconds
    confidence: str                 # "normal" | "low-confidence" | "failed"
    diagnostics: dict               # Method-specific diagnostic quantities
    method_name: str                # Identifier for the estimator


class BaseEstimator:
    """Abstract estimator interface.

    Concrete estimators (physics, change-point, learned) implement this
    contract in later tasks. The methods here only declare the interface.
    """

    def estimate(self, flight: "FlightRecord") -> TDEstimate:
        """Produce a touchdown time estimate for a single flight."""
        raise NotImplementedError

    def name(self) -> str:
        """Return the estimator's identifier."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Runway reference
# ---------------------------------------------------------------------------


@dataclass
class RunwayReference:
    """Runway geometry for projection. All vertical values resolved to HAE internally."""

    threshold_lat: float            # Decimal degrees (>=6 decimal places)
    threshold_lon: float            # Decimal degrees
    heading_deg: float              # Degrees true north (0-360)
    elevation_m: float              # Meters (datum given by elevation_datum)
    elevation_datum: str            # "HAE" | "MSL" - if MSL, geoid-corrected to HAE before any crossing
    geoid_undulation_m: float       # EGM2008 undulation at threshold; added to MSL elevation to get HAE
    length_m: float                 # Meters
    width_m: float                  # Meters (used for wrong-runway lateral-offset gate)
    displaced: bool                 # Whether this is a displaced threshold


# ---------------------------------------------------------------------------
# Flight record (pipeline internal)
# ---------------------------------------------------------------------------


@dataclass
class FlightRecord:
    """Aligned per-flight data record passed to estimators.

    Async position/velocity timestamps are preserved (never merged) for Aireon;
    co-timed sources share the same values across the two time arrays. Velocity
    fields use the native ADS-B units (knots, degrees true, ft/min) as ingested;
    derived signals populated by Module 3 are SI.
    """

    flight_id: str
    aircraft_type: str              # ICAO type designator
    ads_b_source: str               # "aireon" | "flightradar24"

    # Raw timestamps (preserved async for Aireon)
    position_times: np.ndarray      # Epoch seconds for position messages
    velocity_times: np.ndarray      # Epoch seconds for velocity messages

    # Position
    latitudes: np.ndarray           # Decimal degrees
    longitudes: np.ndarray          # Decimal degrees
    geometric_altitudes: np.ndarray   # Meters above WGS-84 (HAE)
    barometric_altitudes: np.ndarray  # Meters (QNH-sensitive)

    # Velocity
    groundspeeds: np.ndarray        # Knots
    tracks: np.ndarray              # Degrees true
    baro_vertical_rates: np.ndarray  # ft/min (may contain NaN)

    # Flags
    on_ground_flags: np.ndarray     # Boolean per position message
    on_ground_transition_time: Optional[float]  # Epoch seconds

    # Runway geometry
    runway: RunwayReference

    # Derived signals (populated by Module 3); SI units
    smoothed_deceleration: Optional[np.ndarray] = None   # m/s^2
    smoothed_jerk: Optional[np.ndarray] = None           # m/s^3
    derivative_uncertainties: Optional[np.ndarray] = None  # 1-sigma of derivatives
    distance_to_threshold: Optional[np.ndarray] = None   # meters
    time_deltas: Optional[np.ndarray] = None             # seconds (irregular-spacing channel)


# ---------------------------------------------------------------------------
# Fusion layer interface
# ---------------------------------------------------------------------------


@dataclass
class FusedEstimate:
    """Output of the fusion/ensemble layer."""

    t_td: float                     # Fused touchdown time (epoch seconds)
    sigma_t: float                  # Fused 1-sigma uncertainty (seconds)
    ci_90_lower: float              # 90% CI lower bound (seconds)
    ci_90_upper: float              # 90% CI upper bound (seconds)
    confidence: str                 # "normal" | "low-confidence" | "no-estimate"
    reason_code: Optional[str]      # Reason for low-confidence/no-estimate
    contributing_estimators: list   # Names of estimators that contributed
    excluded_estimators: list       # Names excluded (with reasons)
    per_estimator_results: dict     # {method_name: TDEstimate} for traceability


class FusionEnsemble:
    """Abstract fusion interface; implemented by later tasks."""

    def fuse(self, estimates: list[TDEstimate], context: "FlightRecord") -> FusedEstimate:
        """Combine estimator outputs into a calibrated fused estimate."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Output record (output boundary; feet/knots)
# ---------------------------------------------------------------------------


@dataclass
class TouchdownResult:
    """Final output record per flight.

    This is the single output boundary where SI internal quantities are
    converted to presentation units. Fields ending in ``_ft`` are feet and
    fields ending in ``_kt`` are knots; ``touchdown_time`` and CI time bounds
    remain in seconds.
    """

    flight_id: str
    aircraft_type: str
    ads_b_source: str

    # Primary outputs
    touchdown_time: float                   # Epoch seconds
    along_runway_distance_ft: float         # Feet from threshold
    lateral_offset_ft: float                # Feet from centerline
    groundspeed_at_touchdown_kt: float      # Knots

    # Uncertainty
    time_ci_90_lower: float                 # Seconds
    time_ci_90_upper: float                 # Seconds
    distance_ci_90_lower_ft: float          # Feet
    distance_ci_90_upper_ft: float          # Feet
    speed_ci_90_lower_kt: float             # Knots
    speed_ci_90_upper_kt: float             # Knots

    # Classification & confidence
    trajectory_type: str                    # "completed-landing" | "go-around" | "touch-and-go"
    confidence: str                         # "normal" | "low-confidence" | "no-estimate"
    reason_code: Optional[str]

    # Diagnostics
    contributing_estimators: list[str]
    excluded_estimators: list[str]
    physics_anchor_t_td: Optional[float]    # Always included per Req 6 (epoch seconds)
    physics_anchor_diagnostics: Optional[dict]
    lever_arm_used: LeverArm
    lever_arm_missing: bool                  # True -> class-median default applied, CI widened
    assumed_touchdown_pitch_deg: float       # Pitch is assumed, not measured (degrees)
    geometric_altitude_available: bool       # Whether vertical estimators could run for this source
    runway_elevation_datum: str              # "HAE" | "MSL"; geoid undulation applied if MSL
    suspected_wrong_runway: bool             # Lateral offset exceeded half-width + margin
    out_of_bounds: bool                      # Distance > runway length or < 0

    # Provenance
    data_version: str
    code_commit: str
    config_hash: str
    model_artifact_hash: Optional[str]


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


@dataclass
class AireonMessage:
    """Per-flight ADS-B input record (Aireon format - async timestamps).

    Position fields are populated when ``message_type == "position"``; velocity
    fields when ``message_type == "velocity"``.
    """

    flight_id: str
    message_type: str               # "position" | "velocity"
    timestamp: float                # Emission time (epoch seconds)

    # Position fields (when message_type == "position")
    latitude: Optional[float] = None            # Decimal degrees
    longitude: Optional[float] = None           # Decimal degrees
    geometric_altitude_m: Optional[float] = None  # Meters above WGS-84 (HAE)
    barometric_altitude_m: Optional[float] = None  # Meters (QNH-sensitive)
    on_ground: Optional[bool] = None

    # Velocity fields (when message_type == "velocity")
    groundspeed_kt: Optional[float] = None      # Knots
    track_deg: Optional[float] = None           # Degrees true
    baro_vertical_rate_ftmin: Optional[float] = None  # ft/min


@dataclass
class FR24Record:
    """Per-flight ADS-B input record (FlightRadar24 format - co-timed).

    ASSUMPTION (unconfirmed): ``altitude_m`` is BAROMETRIC (pressure) altitude,
    not geometric (HAE), and samples are provider-INTERPOLATED rather than raw.
    Until confirmed, geometric/vertical estimators are disabled for this source
    and samples are not treated as independent observations. This is driven by
    :class:`SourceCapability` so it can be flipped once confirmed.
    """

    flight_id: str
    timestamp: float                # Single timestamp for all fields (epoch seconds)
    latitude: float                 # Decimal degrees
    longitude: float                # Decimal degrees
    altitude_m: float               # Meters; assumed barometric - confirm provenance
    altitude_kind: str              # "barometric" | "geometric" | "unknown"
    groundspeed_kt: float           # Knots
    track_deg: float                # Degrees true
    on_ground: bool
    vertical_rate_ftmin: Optional[float] = None  # ft/min


# ---------------------------------------------------------------------------
# QAR truth & validation metrics
# ---------------------------------------------------------------------------


@dataclass
class QARTruthRecord:
    """Quick Access Recorder ground-truth touchdown record."""

    flight_id: str
    touchdown_time_qar: float       # QAR clock (epoch seconds)
    touchdown_lat: float            # Decimal degrees
    touchdown_lon: float            # Decimal degrees
    clock_offset_estimate: Optional[float]  # QAR - ADS-B offset (seconds)
    clock_offset_quality: str       # "good" | "degraded" | "failed"
    aircraft_type: str              # ICAO type designator
    runway_id: str
    airport_id: str
    tail_number: str


@dataclass
class ValidationMetrics:
    """Metrics computed per stratum or overall."""

    n_flights: int

    # Distance error (feet)
    distance_rmse_ft: float
    distance_median_abs_error_ft: float
    distance_iqr_ft: tuple[float, float]
    distance_p95_abs_error_ft: float
    distance_p99_abs_error_ft: float
    distance_p95_long_side_ft: float    # 95th percentile of positive (long-side) errors
    distance_median_signed_error_ft: float

    # Time error (seconds)
    time_rmse_s: float
    time_median_abs_error_s: float

    # Baseline comparison
    baseline_rmse_ft: float
    improvement_pct: float              # (baseline - system) / baseline * 100

    # Coverage
    ci_90_coverage: float               # Fraction of true values inside 90% CI

    # Metadata
    stratum_key: Optional[str] = None   # e.g. "B738", "aireon", "KJFK/04L"


# ---------------------------------------------------------------------------
# Failure reason codes
# ---------------------------------------------------------------------------


class FailureReason(Enum):
    """No-estimate and low-confidence reason codes.

    Note: the design text lists ``MISSING_LEVER_ARM`` twice (once under the
    initial low-confidence list and again with the class-median wording). It is
    defined once here; the class-median-default behavior is the authoritative
    meaning (default lever arm applied, distance CI widened).
    """

    # No-estimate reasons
    INSUFFICIENT_SAMPLES = "insufficient_samples"       # <3 valid samples near t_td
    NO_GROUNDSPEED = "no_groundspeed_data"              # Missing groundspeed entirely
    GAP_SPANS_TOUCHDOWN = "gap_spans_touchdown"         # >15s continuous gap over t_td
    EXCESSIVE_EXCLUSIONS = "excessive_exclusions"       # >50% samples excluded by QA
    ALL_ESTIMATORS_FAILED = "all_estimators_failed"     # Every estimator reported failure
    INVALID_RUNWAY_REF = "invalid_runway_reference"     # Missing/invalid runway geometry
    GO_AROUND = "go_around"                             # No touchdown occurred (approach + climb-out)
    TOUCH_AND_GO = "touch_and_go"                       # Brief contact, no sustained ground roll

    # Low-confidence reasons
    SPARSE_NEAR_TD = "sparse_near_touchdown"            # <3 samples within 30s but >=3 within 60s
    WIDE_CONFIDENCE_INTERVAL = "wide_ci"                # 90% CI width > 600 ft (configurable)
    MISSING_VERTICAL_RATE = "missing_vertical_rate"     # Baro VR null; subset of estimators used
    # MISSING_LEVER_ARM: class-median default applied; distance CI widened.
    # (Design listed this code twice; deduplicated to a single member here.)
    MISSING_LEVER_ARM = "missing_lever_arm"
    ESTIMATOR_DISAGREEMENT = "estimator_disagreement"   # High variance across estimators
    OUT_OF_BOUNDS_POSITION = "out_of_bounds_position"   # Distance exceeds runway or is negative
    DEGRADED_INTERPOLATION = "degraded_interpolation"   # Velocity missing for kinematic interp
    INSUFFICIENT_FLARE_SAMPLES = "insufficient_flare"   # <3 samples in extended vertical fit region
    NO_GROUND_ROLL_CONFIRMATION = "no_ground_roll"      # <2 position samples after on-ground
    GEOMETRIC_ALT_UNAVAILABLE = "geometric_alt_unavailable"  # Source lacks HAE; vertical estimators disabled
    SUSPECTED_WRONG_RUNWAY = "suspected_wrong_runway"   # Lateral offset exceeds half-width + margin
    CLOCK_OFFSET_EXCEEDED = "clock_offset_exceeded"     # Excluded from time-domain training/validation
    DATUM_UNRESOLVED = "datum_unresolved"               # Runway elevation datum/geoid could not be resolved
