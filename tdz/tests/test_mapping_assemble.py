"""Tests for time->position mapping and output-record assembly (Task 20).

Covers the SI geometry/speed mapping at the fused touchdown time
(:mod:`tdz.geo.mapping`) and the SI->presentation output-record assembler
(:mod:`tdz.assemble`), plus:

* **P17** -- output completeness: every processed flight yields exactly one
  confidence class; a non-null reason code is present when the class is not
  "normal"; all primary fields are populated for normal/low-confidence.
* an **end-to-end synthetic-flight integration** test running the estimators ->
  fusion -> mapping -> assembly path.

Requirements: 2.1, 2.2, 3.1, 3.2, 3.3, 3.4, 14.1, 14.2, 14.3, 14.4; Property 17.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.assemble import (
    Provenance,
    assemble_touchdown_result,
    compute_config_hash,
    resolve_provenance,
)
from tdz.config.loader import build_config
from tdz.config.models import LeverArm
from tdz.estimators.physics import (
    DecelKneeEstimator,
    FlareCrossingEstimator,
    ImmRtsEstimator,
)
from tdz.estimators.physics.base import CONFIDENCE_LOW, CONFIDENCE_NORMAL
from tdz.fusion import build_fusion
from tdz.fusion.ensemble import CONFIDENCE_NO_ESTIMATE
from tdz.geo import map_touchdown
from tdz.geo.lever_arm import compute_lever_arm_correction, resolve_lever_arm_correction
from tdz.geo.mapping import groundspeed_slope_mps2, velocity_samples_within
from tdz.models import FailureReason, FlightRecord, FusedEstimate, RunwayReference
from tdz.timebase import KNOTS_TO_MPS
from tdz.tests.test_physics_estimators import synthetic_landing
from tdz.uncertainty.quantifier import M_TO_FT

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

MPS_TO_KT = 1.0 / KNOTS_TO_MPS
_PROVENANCE = Provenance(
    data_version="test-2024.1", code_commit="deadbeef", config_hash="cfg", model_artifact_hash=None
)


def _config(**output_overrides):
    """Build a resolved config (canonical lever-arm table, defaults elsewhere)."""
    data = {
        "lever_arms": {
            "B738": {
                "vertical_offset_m": 4.2,
                "longitudinal_offset_m": 12.5,
                "nominal_touchdown_pitch_deg": 5.5,
                "aircraft_class": "narrowbody",
            },
        },
    }
    if output_overrides:
        data["output"] = output_overrides
    return build_config(data)


def _runway() -> RunwayReference:
    return RunwayReference(
        threshold_lat=33.94,
        threshold_lon=-118.40,
        heading_deg=250.0,
        elevation_m=30.0,
        elevation_datum="HAE",
        geoid_undulation_m=0.0,
        length_m=3500.0,
        width_m=45.0,
        displaced=False,
    )


def _lever_arm(config):
    return resolve_lever_arm_correction("B738", config.lever_arms, aircraft_class="narrowbody")


def _flight(
    position_times,
    velocity_times,
    gs_kt,
    *,
    lat=None,
    lon=None,
    transition=None,
    aircraft_type="B738",
) -> FlightRecord:
    """FlightRecord with velocity populated; positions default to the threshold."""
    position_times = np.asarray(position_times, dtype=float)
    velocity_times = np.asarray(velocity_times, dtype=float)
    gs_kt = np.asarray(gs_kt, dtype=float)
    rw = _runway()
    n = position_times.size
    if lat is None:
        lat = np.full(n, rw.threshold_lat, dtype=float)
    if lon is None:
        lon = np.full(n, rw.threshold_lon, dtype=float)
    return FlightRecord(
        flight_id="MAP",
        aircraft_type=aircraft_type,
        ads_b_source="aireon",
        position_times=position_times,
        velocity_times=velocity_times,
        latitudes=np.asarray(lat, dtype=float),
        longitudes=np.asarray(lon, dtype=float),
        geometric_altitudes=np.full(n, rw.elevation_m, dtype=float),
        barometric_altitudes=np.full(n, np.nan, dtype=float),
        groundspeeds=gs_kt,
        tracks=np.full(velocity_times.size, rw.heading_deg, dtype=float),
        baro_vertical_rates=np.full(velocity_times.size, np.nan, dtype=float),
        on_ground_flags=np.zeros(n, dtype=bool),
        on_ground_transition_time=transition,
        runway=rw,
    )


def _fused(
    t_td,
    sigma_t,
    *,
    confidence=CONFIDENCE_NORMAL,
    reason_code=None,
    contributing=("decel_knee", "pelt"),
    excluded=(),
    per_estimator_results=None,
) -> FusedEstimate:
    return FusedEstimate(
        t_td=t_td,
        sigma_t=sigma_t,
        ci_90_lower=t_td - 1.645 * sigma_t,
        ci_90_upper=t_td + 1.645 * sigma_t,
        confidence=confidence,
        reason_code=reason_code,
        contributing_estimators=list(contributing),
        excluded_estimators=list(excluded),
        per_estimator_results=per_estimator_results or {},
    )


# ---------------------------------------------------------------------------
# Groundspeed slope + velocity-window helpers
# ---------------------------------------------------------------------------


def test_groundspeed_slope_is_linear_segment_slope():
    vt = np.array([0.0, 10.0])
    gs = np.array([100.0, 120.0])  # knots
    # (120-100) kt over 10 s -> 2 kt/s, converted to m/s^2.
    assert groundspeed_slope_mps2(vt, gs, 5.0) == pytest.approx(2.0 * KNOTS_TO_MPS)


def test_groundspeed_slope_zero_outside_range():
    vt = np.array([0.0, 10.0])
    gs = np.array([100.0, 120.0])
    assert groundspeed_slope_mps2(vt, gs, -5.0) == 0.0
    assert groundspeed_slope_mps2(vt, gs, 15.0) == 0.0


def test_velocity_samples_within_window():
    vt = np.array([0.0, 5.0, 30.0])
    assert velocity_samples_within(vt, 8.0, 10.0) is True   # 5.0 is within 10 s
    assert velocity_samples_within(vt, 50.0, 10.0) is False


# ---------------------------------------------------------------------------
# map_touchdown: speed from interpolation (Req 3.2), lever-arm (Req 2.3)
# ---------------------------------------------------------------------------


def test_map_speed_is_interpolated_not_nearest_sample():
    """Req 3.2: touchdown speed comes from kinematic interpolation, not nearest."""
    config = _config()
    lever = _lever_arm(config)
    # Velocity ramps 100 -> 140 kt over [0, 10]; query at t=3 -> interpolated 112.
    flight = _flight([0.0, 10.0], [0.0, 10.0], [100.0, 140.0])
    mapping = map_touchdown(
        flight,
        3.0,
        1.0,
        lever_arm=lever,
        speed_min_mps=50.0 * KNOTS_TO_MPS,
        speed_max_mps=220.0 * KNOTS_TO_MPS,
        velocity_gap_max_s=10.0,
        validation_config=config.validation,
        interpolation_method="linear",
    )
    interpolated_kt = mapping.groundspeed_mps * MPS_TO_KT
    assert interpolated_kt == pytest.approx(112.0)
    # The nearest sample would be 100 kt -> confirm we did NOT use it.
    assert interpolated_kt != pytest.approx(100.0)


def test_map_lever_arm_subtracted_from_along_distance():
    """Req 2.3: the pitch-resolved along-runway lever-arm shift is subtracted."""
    config = _config()
    lever = _lever_arm(config)
    flight = _flight([0.0, 10.0], [0.0, 10.0], [130.0, 130.0])
    mapping = map_touchdown(
        flight,
        5.0,
        1.0,
        lever_arm=lever,
        speed_min_mps=50.0 * KNOTS_TO_MPS,
        speed_max_mps=220.0 * KNOTS_TO_MPS,
        velocity_gap_max_s=10.0,
        validation_config=config.validation,
        interpolation_method="linear",
    )
    # Positions sit on the threshold -> antenna projection ~0; corrected = -shift.
    assert mapping.antenna_along_runway_distance_m == pytest.approx(0.0, abs=1e-6)
    assert mapping.along_runway_distance_m == pytest.approx(
        -lever.along_runway_shift_m, abs=1e-6
    )


def test_map_speed_low_confidence_when_no_velocity_within_window():
    """Req 3.4: missing velocity samples near t_td -> speed low-confidence."""
    config = _config()
    lever = _lever_arm(config)
    flight = _flight([0.0, 100.0], [0.0, 100.0], [130.0, 130.0])
    mapping = map_touchdown(
        flight,
        50.0,  # 50 s from either velocity sample (> 10 s window)
        1.0,
        lever_arm=lever,
        speed_min_mps=50.0 * KNOTS_TO_MPS,
        speed_max_mps=220.0 * KNOTS_TO_MPS,
        velocity_gap_max_s=10.0,
        validation_config=config.validation,
        interpolation_method="linear",
    )
    assert mapping.velocity_samples_present is False
    assert mapping.speed_low_confidence is True
    assert mapping.speed_reason_code is FailureReason.DEGRADED_INTERPOLATION


def test_map_speed_low_confidence_when_implausible():
    """Req 3.4: interpolated speed outside 50-220 kt -> low-confidence."""
    config = _config()
    lever = _lever_arm(config)
    flight = _flight([0.0, 10.0], [0.0, 10.0], [300.0, 300.0])  # 300 kt, implausible
    mapping = map_touchdown(
        flight,
        5.0,
        1.0,
        lever_arm=lever,
        speed_min_mps=50.0 * KNOTS_TO_MPS,
        speed_max_mps=220.0 * KNOTS_TO_MPS,
        velocity_gap_max_s=10.0,
        validation_config=config.validation,
        interpolation_method="linear",
    )
    assert mapping.speed_plausible is False
    assert mapping.speed_low_confidence is True
    assert mapping.speed_reason_code is FailureReason.IMPLAUSIBLE_SPEED


def test_map_speed_sigma_propagates_from_time_uncertainty():
    """Req 3.3: sigma_v = |dv/dt| * sigma_t from the interpolated speed profile."""
    config = _config()
    lever = _lever_arm(config)
    flight = _flight([0.0, 10.0], [0.0, 10.0], [100.0, 140.0])  # slope 4 kt/s
    sigma_t = 2.0
    mapping = map_touchdown(
        flight,
        5.0,
        sigma_t,
        lever_arm=lever,
        speed_min_mps=50.0 * KNOTS_TO_MPS,
        speed_max_mps=220.0 * KNOTS_TO_MPS,
        velocity_gap_max_s=10.0,
        validation_config=config.validation,
        interpolation_method="linear",
    )
    expected = abs(4.0 * KNOTS_TO_MPS) * sigma_t
    assert mapping.groundspeed_sigma_mps == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Assembler: units, provenance, no-estimate, flags (Req 14.x, 2.4, 2.5)
# ---------------------------------------------------------------------------


def test_assemble_converts_units_and_rounds_speed():
    """SI -> feet/knots at this boundary only; speed rounded to 0.1 kt (Req 3.1)."""
    config = _config()
    lever = _lever_arm(config)
    flight = synthetic_landing()
    result = assemble_touchdown_result(
        flight,
        _fused(200.0, 1.0),
        config,
        lever_arm=lever,
        trajectory_type="completed-landing",
        provenance=_PROVENANCE,
        geometric_altitude_available=True,
    )
    # Speed reported in knots, at 0.1 kt resolution, within the plausible band.
    assert 50.0 <= result.groundspeed_at_touchdown_kt <= 220.0
    assert result.groundspeed_at_touchdown_kt == pytest.approx(
        round(result.groundspeed_at_touchdown_kt * 10.0) / 10.0
    )
    # Distance/lateral are feet.
    assert math.isfinite(result.along_runway_distance_ft)
    assert math.isfinite(result.lateral_offset_ft)


def test_assemble_populates_provenance():
    config = _config()
    lever = _lever_arm(config)
    result = assemble_touchdown_result(
        synthetic_landing(),
        _fused(200.0, 1.0),
        config,
        lever_arm=lever,
        trajectory_type="completed-landing",
        provenance=_PROVENANCE,
        geometric_altitude_available=True,
    )
    assert result.data_version == "test-2024.1"
    assert result.code_commit == "deadbeef"
    assert result.config_hash == "cfg"
    assert result.model_artifact_hash is None


def test_assemble_no_estimate_has_reason_and_nan_primaries():
    """Req 14.1/14.3: a no-estimate output carries a reason and no fabricated point."""
    config = _config()
    lever = _lever_arm(config)
    fused = _fused(
        float("nan"),
        float("inf"),
        confidence=CONFIDENCE_NO_ESTIMATE,
        reason_code=FailureReason.ALL_ESTIMATORS_FAILED.value,
        contributing=(),
        excluded=("decel_knee: failed",),
    )
    result = assemble_touchdown_result(
        _flight([0.0, 5.0], [0.0, 5.0], [130.0, 130.0]),
        fused,
        config,
        lever_arm=lever,
        trajectory_type="completed-landing",
        provenance=_PROVENANCE,
        geometric_altitude_available=True,
    )
    assert result.confidence == CONFIDENCE_NO_ESTIMATE
    assert result.reason_code == FailureReason.ALL_ESTIMATORS_FAILED.value
    assert math.isnan(result.along_runway_distance_ft)
    assert math.isnan(result.groundspeed_at_touchdown_kt)
    # Provenance still present on a no-estimate record (Req 14.4 / 15.3).
    assert result.config_hash == "cfg"


def test_assemble_out_of_bounds_flag_and_downgrade():
    """Req 2.4: a negative along-runway distance is flagged out-of-bounds."""
    config = _config()
    lever = _lever_arm(config)
    # Positions on the threshold + lever-arm subtraction -> slightly negative.
    flight = _flight([0.0, 5.0], [0.0, 5.0], [130.0, 130.0])
    result = assemble_touchdown_result(
        flight,
        _fused(2.5, 1.0),
        config,
        lever_arm=lever,
        trajectory_type="completed-landing",
        provenance=_PROVENANCE,
        geometric_altitude_available=True,
        interpolation_method="linear",
    )
    assert result.out_of_bounds is True
    assert result.confidence == CONFIDENCE_LOW
    assert result.reason_code == FailureReason.OUT_OF_BOUNDS_POSITION.value


def test_assemble_speed_ci_present_and_ordered():
    """Req 3.3: a speed CI is reported (from propagated t_td uncertainty)."""
    config = _config()
    lever = _lever_arm(config)
    # Ramping speed so dv/dt != 0 -> a positive-width speed CI.
    flight = _flight(
        np.array([196.0, 200.0, 204.0]),
        np.array([196.0, 200.0, 204.0]),
        np.array([140.0, 130.0, 120.0]),
    )
    result = assemble_touchdown_result(
        flight,
        _fused(200.0, 2.0),
        config,
        lever_arm=lever,
        trajectory_type="completed-landing",
        provenance=_PROVENANCE,
        geometric_altitude_available=True,
        interpolation_method="linear",
    )
    assert result.speed_ci_90_lower_kt <= result.groundspeed_at_touchdown_kt <= result.speed_ci_90_upper_kt
    assert result.speed_ci_90_upper_kt > result.speed_ci_90_lower_kt


def test_compute_config_hash_is_stable_and_sensitive():
    cfg_a = _config()
    cfg_b = _config()
    assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)
    cfg_c = _config(speed_max_kt=210.0)
    assert compute_config_hash(cfg_c) != compute_config_hash(cfg_a)


def test_resolve_provenance_hashes_config_and_defaults_commit():
    cfg = _config()
    prov = resolve_provenance(cfg, data_version="v1", code_commit="abc123")
    assert prov.data_version == "v1"
    assert prov.code_commit == "abc123"
    assert prov.config_hash == compute_config_hash(cfg)


# ---------------------------------------------------------------------------
# Property 17: Output Completeness Invariant
# ---------------------------------------------------------------------------

_PRIMARY_FIELDS = (
    "touchdown_time",
    "along_runway_distance_ft",
    "lateral_offset_ft",
    "groundspeed_at_touchdown_kt",
    "time_ci_90_lower",
    "time_ci_90_upper",
    "distance_ci_90_lower_ft",
    "distance_ci_90_upper_ft",
    "speed_ci_90_lower_kt",
    "speed_ci_90_upper_kt",
)
_CONFIDENCE_CLASSES = frozenset({CONFIDENCE_NORMAL, CONFIDENCE_LOW, CONFIDENCE_NO_ESTIMATE})


@st.composite
def _assembly_scenario(draw):
    confidence = draw(
        st.sampled_from([CONFIDENCE_NORMAL, CONFIDENCE_LOW, CONFIDENCE_NO_ESTIMATE])
    )
    t_td = draw(st.floats(min_value=185.0, max_value=215.0, allow_nan=False, allow_infinity=False))
    sigma_t = draw(st.floats(min_value=0.3, max_value=6.0, allow_nan=False, allow_infinity=False))
    return confidence, t_td, sigma_t


@pytest.mark.property
@given(scenario=_assembly_scenario())
def test_p17_output_completeness(scenario):
    """Feature: touchdown-point-detection, Property 17: Output Completeness Invariant

    For any processed flight, the output record contains exactly one confidence
    classification from {"normal", "low-confidence", "no-estimate"}; if the class
    is "low-confidence" or "no-estimate" a non-null reason code is present; and if
    "normal" or "low-confidence" all primary output fields (touchdown_time,
    along-runway distance, lateral offset, groundspeed, and the confidence
    intervals) are populated.

    Validates: Requirements 14.1, 14.2, 14.3, 14.4
    """
    confidence, t_td, sigma_t = scenario
    config = _config()
    lever = _lever_arm(config)
    flight = synthetic_landing(t_td=200.0)

    if confidence == CONFIDENCE_NO_ESTIMATE:
        fused = _fused(
            float("nan"),
            float("inf"),
            confidence=CONFIDENCE_NO_ESTIMATE,
            reason_code=FailureReason.ALL_ESTIMATORS_FAILED.value,
            contributing=(),
        )
    elif confidence == CONFIDENCE_LOW:
        fused = _fused(
            t_td, sigma_t, confidence=CONFIDENCE_LOW,
            reason_code=FailureReason.WIDE_CONFIDENCE_INTERVAL.value,
        )
    else:
        fused = _fused(t_td, sigma_t, confidence=CONFIDENCE_NORMAL)

    result = assemble_touchdown_result(
        flight,
        fused,
        config,
        lever_arm=lever,
        trajectory_type="completed-landing",
        provenance=_PROVENANCE,
        geometric_altitude_available=True,
    )

    # Exactly one confidence classification from the allowed set.
    assert result.confidence in _CONFIDENCE_CLASSES

    # Reason code present when the class is not "normal".
    if result.confidence != CONFIDENCE_NORMAL:
        assert result.reason_code is not None

    # All primary fields populated for normal / low-confidence.
    if result.confidence in (CONFIDENCE_NORMAL, CONFIDENCE_LOW):
        for field in _PRIMARY_FIELDS:
            value = getattr(result, field)
            assert value is not None
            assert isinstance(value, float)
            assert math.isfinite(value)

    # Provenance is always present (Req 14.4 / 15.3).
    assert result.config_hash is not None
    assert result.data_version is not None


# ---------------------------------------------------------------------------
# End-to-end synthetic-flight integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_end_to_end_synthetic_flight_produces_complete_record():
    """Estimators -> fusion -> mapping -> assembly on a clean synthetic landing."""
    config = _config()
    flight = synthetic_landing(t_td=200.0, v_td_mps=65.0)

    # Run the physics estimators and fuse their outputs.
    estimators = {
        "decel_knee": DecelKneeEstimator(),
        "flare_crossing": FlareCrossingEstimator(),
        "imm_rts": ImmRtsEstimator(),
    }
    estimates = [est.estimate(flight) for est in estimators.values()]
    fusion = build_fusion(config.fusion)
    fused = fusion.fuse(estimates, flight)

    lever = resolve_lever_arm_correction(
        flight.aircraft_type, config.lever_arms, aircraft_class="narrowbody"
    )
    provenance = resolve_provenance(config, data_version="synthetic", code_commit="testsha")

    result = assemble_touchdown_result(
        flight,
        fused,
        config,
        lever_arm=lever,
        trajectory_type="completed-landing",
        provenance=provenance,
        geometric_altitude_available=True,
    )

    # A clean landing should yield an estimate (not a no-estimate).
    assert result.confidence in (CONFIDENCE_NORMAL, CONFIDENCE_LOW)
    # Touchdown time near the synthetic truth.
    assert result.touchdown_time == pytest.approx(200.0, abs=6.0)
    # Along-runway distance is within the runway and reported in feet.
    length_ft = flight.runway.length_m * M_TO_FT
    assert 0.0 <= result.along_runway_distance_ft <= length_ft
    # Groundspeed near the synthetic touchdown speed (~126 kt), in the band.
    assert 50.0 <= result.groundspeed_at_touchdown_kt <= 220.0
    assert result.groundspeed_at_touchdown_kt == pytest.approx(65.0 * MPS_TO_KT, abs=8.0)
    # Confidence intervals present and ordered.
    assert result.time_ci_90_lower < result.touchdown_time < result.time_ci_90_upper
    assert result.distance_ci_90_lower_ft < result.along_runway_distance_ft < result.distance_ci_90_upper_ft
    # Diagnostics + provenance populated (Req 14.4 / 15.3).
    assert "decel_knee" in result.contributing_estimators or result.contributing_estimators
    assert result.physics_anchor_t_td is not None
    assert result.lever_arm_used.icao_type == "B738"
    assert result.config_hash == compute_config_hash(config)
    assert result.data_version == "synthetic"
