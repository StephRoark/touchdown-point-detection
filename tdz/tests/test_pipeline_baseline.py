"""Tests for the stage-1-3 baseline wiring (Task 14).

Covers the integration runner :func:`tdz.pipeline.run_stage123`, the naive
"first on-ground sample position" baseline (Req 12.8), the provisional
inverse-variance combiner (explicitly NOT the Task-18 fusion), and the
preliminary along-runway distance-error harness that surfaces the
cadence-limited floor (Req 13.0).

The synthetic-landing generator is adapted from
``tdz/tests/test_physics_estimators.py``: a constant ~3 deg glideslope, a
quadratic flare flattening to the runway at ``t_td``, and a constant-decel
ground roll sampled at the 4-5 s ADS-B cadence. By construction the aircraft is
exactly at the runway threshold at the true ``t_td``, so the truth touchdown
lat/long is the threshold (the centerline point at ``t_td``) and the truth
along-runway distance is ~0 m.

Tolerances are kept honest to the cadence: a touchdown cannot be located more
finely than ~1 sample interval at 4-5 s, so example tolerances are stated as
multiples of the cadence.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from pyproj import Geod

from tdz.config.loader import load_config
from tdz.config.schema import TDZConfig
from tdz.models import FailureReason, FlightRecord, RunwayReference
from tdz.pipeline import (
    BaselineComparison,
    CombinedEstimate,
    FlightTruth,
    NaiveBaselineResult,
    StageRunResult,
    along_runway_truth_distance,
    combine_estimates,
    distance_error_summary,
    naive_baseline_distance,
    run_stage123,
)
from tdz.timebase.interpolation import KNOTS_TO_MPS

_GEOD = Geod(ellps="WGS84")
FT_TO_M = 0.3048

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "tdz_config.yaml"


@pytest.fixture(scope="module")
def config() -> TDZConfig:
    """The canonical repo configuration (physics + change-point estimators)."""
    return load_config(str(_CONFIG_PATH))


# ---------------------------------------------------------------------------
# Synthetic landing helpers (adapted from test_physics_estimators.py)
# ---------------------------------------------------------------------------


def _runway(*, elevation_m: float = 30.0, datum: str = "HAE") -> RunwayReference:
    return RunwayReference(
        threshold_lat=33.94,
        threshold_lon=-118.40,
        heading_deg=250.0,
        elevation_m=elevation_m,
        elevation_datum=datum,
        geoid_undulation_m=0.0,
        length_m=3500.0,
        width_m=45.0,
        displaced=False,
    )


def _approach_latlon(runway: RunwayReference, speed_mps: float, dt_to_td: float):
    """Lat/lon ``dt_to_td`` seconds before threshold (after, when negative)."""
    d = speed_mps * dt_to_td
    if d >= 0.0:
        az = (runway.heading_deg + 180.0) % 360.0
        lon, lat, _ = _GEOD.fwd(runway.threshold_lon, runway.threshold_lat, az, d)
    else:
        lon, lat, _ = _GEOD.fwd(
            runway.threshold_lon, runway.threshold_lat, runway.heading_deg, -d
        )
    return lat, lon


def synthetic_landing(
    *,
    dt: float = 4.5,
    t_td: float = 200.0,
    n_before: int = 12,
    n_after: int = 8,
    v_td_mps: float = 65.0,
    approach_decel: float = 0.5,
    rollout_decel: float = 2.5,
    glide_rate_mps: float = 3.5,
    flare_duration_s: float = 6.0,
    phase_s: float = 0.0,
    velocity_offset_s: float = 0.0,
    omit_geometric: bool = False,
    on_ground_delay_samples: float = 2.0,
    flight_id: str = "SYN",
    ads_b_source: str = "aireon",
) -> FlightRecord:
    """Build a synthetic completed-landing :class:`FlightRecord` (touchdown at threshold)."""
    runway = _runway()
    position_times = phase_s + t_td + np.arange(-n_before, n_after + 1) * dt
    velocity_times = position_times + velocity_offset_s

    h_flare = 50.0 * FT_TO_M
    t_flare = t_td - flare_duration_s

    def height(t: float) -> float:
        if t <= t_flare:
            return h_flare + glide_rate_mps * (t_flare - t)
        if t <= t_td:
            return h_flare * ((t_td - t) / (t_td - t_flare)) ** 2
        return 0.0

    def speed_mps(t: float) -> float:
        if t <= t_td:
            return v_td_mps + approach_decel * (t_td - t)
        return v_td_mps - rollout_decel * (t - t_td)

    heights = np.array([height(float(t)) for t in position_times])
    geo_alt = runway.elevation_m + heights
    if omit_geometric:
        geo_alt = np.full(position_times.size, np.nan)

    v_mps = np.array([max(speed_mps(float(t)), 1.0) for t in velocity_times])
    gs_kt = v_mps / KNOTS_TO_MPS

    lats, lons = [], []
    for t in position_times:
        lat, lon = _approach_latlon(runway, max(speed_mps(float(t)), 1.0), t_td - float(t))
        lats.append(lat)
        lons.append(lon)

    transition = t_td + on_ground_delay_samples * dt
    on_ground_flags = position_times >= transition

    return FlightRecord(
        flight_id=flight_id,
        aircraft_type="B738",
        ads_b_source=ads_b_source,
        position_times=position_times,
        velocity_times=velocity_times,
        latitudes=np.array(lats),
        longitudes=np.array(lons),
        geometric_altitudes=geo_alt,
        barometric_altitudes=np.full(position_times.size, np.nan),
        groundspeeds=gs_kt,
        tracks=np.full(velocity_times.size, runway.heading_deg),
        baro_vertical_rates=np.full(velocity_times.size, np.nan),
        on_ground_flags=on_ground_flags,
        on_ground_transition_time=float(transition),
        runway=runway,
    )


def synthetic_go_around(
    *,
    dt: float = 4.5,
    t_low: float = 200.0,
    n: int = 18,
    min_height_m: float = 25.0,
    flight_id: str = "GOA",
) -> FlightRecord:
    """A descent that levels at ``min_height_m`` (never contacts) and climbs out.

    No sample reaches the contact height and the on-ground flag never sets, so
    the trajectory classifies as a go-around (no contact segment).
    """
    runway = _runway()
    position_times = t_low + np.arange(-n // 2, n - n // 2) * dt

    # V-shaped height profile bottoming at min_height_m at t_low, climbing well
    # above the climb-out height on either side.
    descent_rate = 3.5  # m/s
    heights = min_height_m + descent_rate * np.abs(position_times - t_low)
    geo_alt = runway.elevation_m + heights

    # Speed dips through the approach and recovers (no sustained decel).
    speeds_mps = 75.0 - 5.0 * np.exp(-((position_times - t_low) ** 2) / (2 * 20.0 ** 2))
    gs_kt = speeds_mps / KNOTS_TO_MPS

    lats, lons = [], []
    for t in position_times:
        # Place along the approach centerline (before threshold); never lands.
        lat, lon = _approach_latlon(runway, 70.0, max(t_low - float(t), 0.0) + 5.0)
        lats.append(lat)
        lons.append(lon)

    return FlightRecord(
        flight_id=flight_id,
        aircraft_type="B738",
        ads_b_source="aireon",
        position_times=position_times,
        velocity_times=position_times.copy(),
        latitudes=np.array(lats),
        longitudes=np.array(lons),
        geometric_altitudes=geo_alt,
        barometric_altitudes=np.full(position_times.size, np.nan),
        groundspeeds=gs_kt,
        tracks=np.full(position_times.size, runway.heading_deg),
        baro_vertical_rates=np.full(position_times.size, np.nan),
        on_ground_flags=np.zeros(position_times.size, dtype=bool),
        on_ground_transition_time=None,
        runway=runway,
    )


def _truth_for(flight: FlightRecord, *, t_td: float = 200.0) -> FlightTruth:
    """Truth touchdown = the centerline point at the true ``t_td`` (the threshold)."""
    lat, lon = _approach_latlon(flight.runway, 65.0, 0.0)
    return FlightTruth(flight_id=flight.flight_id, touchdown_lat=lat, touchdown_lon=lon)


# ===========================================================================
# End-to-end stage-1-3 run
# ===========================================================================


def test_run_stage123_clean_landing_combines_near_truth(config: TDZConfig):
    """A clean aireon landing yields a combined t_td near truth and a finite distance.

    The combined t_td is within ~1.5x cadence of the true touchdown and the
    along-runway combined distance is finite and near the threshold (truth ~0 m).
    The estimates dict includes the eligible physics + change-point estimators.
    """
    dt = 4.5
    t_td = 200.0
    flight = synthetic_landing(dt=dt, t_td=t_td)
    result = run_stage123(flight, config)

    assert not result.no_touchdown
    assert result.bracket.status == "ok"
    assert result.combined.ok
    assert result.combined_t_td is not None
    assert abs(result.combined_t_td - t_td) <= 1.5 * dt

    assert result.combined_distance_m is not None
    assert math.isfinite(result.combined_distance_m)
    # Touchdown is at the threshold (truth ~0 m); the combined distance lands
    # within ~1.5 cadence-intervals of along-track travel of it.
    cadence_travel_m = 1.5 * dt * 65.0
    assert abs(result.combined_distance_m) <= cadence_travel_m

    # Eligible estimators (aireon has geometric altitude) were run.
    for name in ("decel_knee", "flare_crossing", "imm_rts", "jerk_onset", "pelt"):
        assert name in result.estimates
    assert len(result.combined.contributing) >= 2


def test_run_stage123_fr24_excludes_geometric_estimators_but_still_combines(
    config: TDZConfig,
):
    """An FR24/no-geometric flight excludes flare_crossing & imm_rts, still combines.

    The velocity-stream estimators (decel_knee / change-point) still produce a
    combined estimate (Req 8.5 / Property 20 gating).
    """
    dt = 4.5
    flight = synthetic_landing(
        dt=dt, omit_geometric=True, ads_b_source="flightradar24", flight_id="FR24"
    )
    result = run_stage123(flight, config)

    assert "flare_crossing" in result.gating.excluded_estimators
    assert "imm_rts" in result.gating.excluded_estimators
    assert "flare_crossing" not in result.estimates
    assert "imm_rts" not in result.estimates
    # Velocity-stream estimators still ran and contributed.
    assert "decel_knee" in result.estimates
    assert result.combined.ok
    assert result.combined_distance_m is not None
    assert math.isfinite(result.combined_distance_m)
    assert abs(result.combined_t_td - 200.0) <= 1.5 * dt


def test_run_stage123_go_around_is_no_touchdown(config: TDZConfig):
    """A go-around surfaces the no-touchdown bracket; no fabricated distance."""
    flight = synthetic_go_around()
    result = run_stage123(flight, config)

    assert result.no_touchdown
    assert result.bracket.status == "no-touchdown"
    assert result.reason_code == FailureReason.GO_AROUND
    assert result.combined_distance_m is None
    assert result.combined_t_td is None
    assert not result.combined.ok


# ===========================================================================
# Naive baseline (Req 12.8)
# ===========================================================================


def test_naive_baseline_is_first_on_ground_sample(config: TDZConfig):
    """The naive baseline projects the first on-ground sample's position."""
    flight = synthetic_landing(on_ground_delay_samples=2.0)
    naive = naive_baseline_distance(flight)

    assert naive.available
    flags = np.asarray(flight.on_ground_flags, dtype=bool)
    expected_idx = int(np.argmax(flags))
    assert naive.sample_index == expected_idx
    assert naive.sample_time == pytest.approx(flight.position_times[expected_idx])
    # The first on-ground sample is well down the runway (positive, past threshold).
    assert naive.distance_m is not None and naive.distance_m > 0.0


def test_naive_baseline_unavailable_without_on_ground_flag():
    """No on-ground sample -> baseline unavailable."""
    flight = synthetic_go_around()
    naive = naive_baseline_distance(flight)
    assert not naive.available
    assert naive.distance_m is None


# ===========================================================================
# System beats the naive strawman on the synthetic set (Req 12.8 / 13.4)
# ===========================================================================


def test_system_beats_naive_baseline_on_synthetic_set(config: TDZConfig):
    """Over a synthetic landing set the system materially outperforms the strawman.

    The on-ground flag lags touchdown, so the naive distance overshoots down the
    runway; the physics-combined estimate stays near the true threshold. The
    BaselineComparison therefore shows a positive (and large) RMSE improvement.
    """
    t_td = 200.0
    flights = [
        synthetic_landing(t_td=t_td, phase_s=phase, v_td_mps=v, flight_id=f"SYN{i}")
        for i, (phase, v) in enumerate(
            [(0.0, 65.0), (1.0, 62.0), (2.2, 70.0), (3.0, 60.0), (1.5, 68.0), (0.5, 64.0)]
        )
    ]
    results = [run_stage123(f, config) for f in flights]
    truths = {f.flight_id: _truth_for(f) for f in flights}

    summary = distance_error_summary(results, truths)

    assert summary.n_flights == len(flights)
    # The system error is much smaller than the naive baseline error.
    assert summary.system_rmse_m < summary.baseline_rmse_m
    assert summary.rmse_improvement_pct > 30.0  # Req 13.4 target direction
    # The naive baseline is systematically long (positive signed error).
    assert summary.baseline_median_signed_error_m > 0.0
    # The preliminary cadence-limited floor equals the system RMSE here.
    assert summary.cadence_limited_floor_m == pytest.approx(summary.system_rmse_m)
    assert math.isfinite(summary.cadence_limited_floor_m)


# ===========================================================================
# Distance-error summary arithmetic (unit check on a tiny known set)
# ===========================================================================


def _fabricated_result(
    flight_id: str,
    runway: RunwayReference,
    *,
    system_distance_m: float,
    naive_distance_m: float,
) -> StageRunResult:
    """A minimal StageRunResult carrying only the fields the summary consumes."""
    naive = NaiveBaselineResult(
        available=True,
        distance_m=naive_distance_m,
        lateral_offset_m=0.0,
        sample_time=10.0,
        sample_index=3,
    )
    combined = CombinedEstimate(
        ok=True,
        t_td=100.0,
        sigma_t=1.0,
        contributing=("decel_knee",),
        weights={"decel_knee": 1.0},
        reason_code=None,
    )
    return StageRunResult(
        flight_id=flight_id,
        ads_b_source="aireon",
        runway=runway,
        bracket=None,  # unused by the summary
        qa_status="ok",
        qa_reason=None,
        gating=None,  # unused by the summary
        estimates={},
        combined=combined,
        combined_t_td=100.0,
        combined_distance_m=system_distance_m,
        lateral_offset_m=0.0,
        naive_baseline=naive,
        no_touchdown=False,
        reason_code=None,
        diagnostics={},
    )


def test_distance_error_summary_arithmetic():
    """RMSE / median / p95 / improvement computed correctly on a known set.

    Truth touchdown is the runway threshold (along-runway truth distance = 0 m),
    so the signed error of each method equals its raw along-runway distance.
    """
    runway = _runway()
    # Truth = threshold => truth along-runway distance is exactly 0 m.
    truth_lat, truth_lon = runway.threshold_lat, runway.threshold_lon
    assert along_runway_truth_distance(runway, truth_lat, truth_lon) == pytest.approx(
        0.0, abs=1e-6
    )

    system = [10.0, -20.0, 30.0]
    naive = [100.0, 120.0, 140.0]
    results = [
        _fabricated_result(
            f"F{i}", runway, system_distance_m=s, naive_distance_m=b
        )
        for i, (s, b) in enumerate(zip(system, naive))
    ]
    truths = {f"F{i}": FlightTruth(f"F{i}", truth_lat, truth_lon) for i in range(3)}

    summary = distance_error_summary(results, truths)

    assert summary.n_flights == 3

    sys_err = np.array(system)
    base_err = np.array(naive)
    expected_sys_rmse = math.sqrt(float(np.mean(sys_err**2)))
    expected_base_rmse = math.sqrt(float(np.mean(base_err**2)))

    assert summary.system_rmse_m == pytest.approx(expected_sys_rmse)
    assert summary.baseline_rmse_m == pytest.approx(expected_base_rmse)
    assert summary.system_median_abs_error_m == pytest.approx(20.0)  # median(|10,20,30|)
    assert summary.baseline_median_abs_error_m == pytest.approx(120.0)
    assert summary.system_median_signed_error_m == pytest.approx(10.0)  # median(10,-20,30)
    assert summary.system_p95_abs_error_m == pytest.approx(
        float(np.percentile(np.abs(sys_err), 95.0))
    )
    expected_improvement = (
        (expected_base_rmse - expected_sys_rmse) / expected_base_rmse * 100.0
    )
    assert summary.rmse_improvement_pct == pytest.approx(expected_improvement)
    assert summary.rmse_improvement_pct > 30.0


# ===========================================================================
# Provisional combiner (NOT the Task-18 fusion)
# ===========================================================================


def test_combine_estimates_is_inverse_variance_weighted_mean():
    """The provisional combiner is a 1/sigma^2-weighted mean of eligible estimates."""
    from tdz.models import TDEstimate

    estimates = {
        "decel_knee": TDEstimate(100.0, 2.0, "normal", {}, "decel_knee"),
        "pelt": TDEstimate(104.0, 1.0, "normal", {}, "pelt"),
        # Failed estimate must be ignored.
        "cusum": TDEstimate(float("nan"), float("inf"), "failed", {}, "cusum"),
        # Ineligible estimate must be ignored even though it is present.
        "flare_crossing": TDEstimate(80.0, 0.5, "normal", {}, "flare_crossing"),
    }
    eligible = ("decel_knee", "pelt", "cusum")  # flare_crossing excluded by gating
    combined = combine_estimates(estimates, eligible)

    assert combined.ok
    assert set(combined.contributing) == {"decel_knee", "pelt"}
    w1, w2 = 1.0 / 4.0, 1.0 / 1.0
    expected = (w1 * 100.0 + w2 * 104.0) / (w1 + w2)
    assert combined.t_td == pytest.approx(expected)
    assert combined.sigma_t == pytest.approx(math.sqrt(1.0 / (w1 + w2)))


def test_combine_estimates_all_failed_returns_no_estimate():
    """When every eligible estimate failed, the combiner reports no estimate."""
    from tdz.models import TDEstimate

    estimates = {
        "decel_knee": TDEstimate(float("nan"), float("inf"), "failed", {}, "decel_knee"),
        "pelt": TDEstimate(float("nan"), float("inf"), "failed", {}, "pelt"),
    }
    combined = combine_estimates(estimates, ("decel_knee", "pelt"))
    assert not combined.ok
    assert combined.reason_code == FailureReason.ALL_ESTIMATORS_FAILED
