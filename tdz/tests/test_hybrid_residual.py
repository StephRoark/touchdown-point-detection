"""Tests for the hybrid-residual estimator (Task 17, optional).

The hybrid-residual model keeps a **physical backbone** (a physics estimator)
and trains a learned model to predict the *residual* of that backbone's
touchdown estimate rather than absolute time (design "Learned Estimators Detail
-> Hybrid Residual (Optional)"; Req 5.3, 6.2). The final estimate reconstructs
the absolute time as ``t_td = physics_anchor_t_td + predicted_residual``.

Coverage
--------
* **Reconstruction (Req 5.3)** -- the reported ``t_td`` equals the physics
  anchor ``t_td`` plus the predicted residual (the defining identity), when the
  on-ground bound does not clamp.
* **Physics anchor carried through (Req 6.2)** -- the anchor's ``t_td``,
  uncertainty, and diagnostics are present in the estimate's diagnostics.
* **Learning** -- after training, the held-out touchdown error beats the raw
  physics-anchor baseline (the learned model corrects a systematic bias).
* **Uncertainty** -- the residual quantile spread yields a positive finite
  ``sigma_t`` with ordered, bracketing bounds.
* **Unfit / failure** -- an untrained estimator returns a *failed* estimate
  (does not raise); a flight whose physics backbone fails fails gracefully with
  the anchor still recorded.
* **On-ground bound** -- the inherited Requirement-18 upper bound still clamps a
  reconstructed ``t_td`` at/after the on-ground transition (Property 5).
* **Reproducibility** -- training twice with the same seed gives identical
  predictions (Req 15.1/15.2).

LightGBM is imported via ``importorskip`` so the suite degrades gracefully if it
is unavailable; it is a declared dependency and expected to be installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.estimators.learned import HybridResidualEstimator
from tdz.estimators.physics import DecelKneeEstimator
from tdz.models import FailureReason, QARTruthRecord
from tdz.tests.test_physics_estimators import synthetic_landing

# LightGBM is a declared dependency; skip gracefully if somehow unavailable.
pytest.importorskip("lightgbm")


# ---------------------------------------------------------------------------
# Synthetic labeled-dataset helpers
# ---------------------------------------------------------------------------


def _truth(flight_id: str, t_td: float, aircraft_type: str = "B738") -> QARTruthRecord:
    """A minimal QAR truth record carrying the known touchdown time."""
    return QARTruthRecord(
        flight_id=flight_id,
        touchdown_time_qar=t_td,
        touchdown_lat=33.94,
        touchdown_lon=-118.40,
        clock_offset_estimate=0.0,
        clock_offset_quality="good",
        aircraft_type=aircraft_type,
        runway_id="04L",
        airport_id="KLAX",
        tail_number="N12345",
    )


def _make_dataset(n: int, seed: int):
    """Build ``n`` synthetic labeled landings with varied dynamics.

    A small, deterministic per-flight bias is injected between the QAR truth and
    the clean two-slope knee so the physics anchor carries a *systematic*
    residual the learned model can learn to correct (mirrors the design intent
    of the hybrid-residual model).
    """
    rng = np.random.default_rng(seed)
    flights = []
    truths = []
    for i in range(n):
        knee_t = 200.0 + float(rng.uniform(-40.0, 40.0))
        flight = synthetic_landing(
            dt=float(rng.uniform(4.0, 5.0)),
            t_td=knee_t,
            n_before=int(rng.integers(10, 16)),
            n_after=int(rng.integers(6, 11)),
            v_td_mps=float(rng.uniform(58.0, 80.0)),
            approach_decel=float(rng.uniform(0.3, 0.9)),
            rollout_decel=float(rng.uniform(1.6, 3.2)),
            on_ground_delay_samples=float(rng.uniform(2.5, 4.0)),
            flight_id=f"F{i:03d}",
        )
        # Systematic bias (+ small noise): truth sits ~1.5 s before the knee.
        truth_t = knee_t - 1.5 + float(rng.normal(0.0, 0.25))
        flights.append(flight)
        truths.append(_truth(f"F{i:03d}", truth_t))
    return flights, truths


@pytest.fixture(scope="module")
def labeled_dataset():
    """A fixed synthetic dataset: 64 landings, split 50 train / 14 held-out."""
    flights, truths = _make_dataset(64, seed=20240717)
    return flights, truths


@pytest.fixture(scope="module")
def trained_estimator(labeled_dataset):
    """A hybrid-residual estimator trained on the first 50 landings."""
    flights, truths = labeled_dataset
    est = HybridResidualEstimator(n_estimators=150, num_leaves=15, seed=12345)
    est.train(flights[:50], truths[:50])
    return est


# ---------------------------------------------------------------------------
# Reconstruction identity (Req 5.3) and anchor carry-through (Req 6.2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_final_t_td_equals_anchor_plus_residual(labeled_dataset, trained_estimator):
    """t_td == physics_anchor_t_td + predicted_residual (the defining identity).

    Checked on held-out flights where the on-ground bound does not clamp, so the
    reconstructed time is reported unchanged.
    """
    flights, _ = labeled_dataset
    checked = 0
    for flight in flights[50:]:
        estimate = trained_estimator.estimate(flight)
        if estimate.diagnostics.get("on_ground_clamped"):
            continue  # clamped: the identity is intentionally overridden by Req 18
        anchor_t_td = estimate.diagnostics["physics_anchor_t_td"]
        residual = estimate.diagnostics["predicted_residual_s"]
        assert estimate.t_td == pytest.approx(anchor_t_td + residual, abs=1e-9)
        checked += 1
    assert checked > 0, "expected at least one unclamped held-out flight"


@pytest.mark.unit
def test_physics_anchor_carried_in_record(trained_estimator):
    """The physics anchor t_td, uncertainty, and diagnostics are in the record (Req 6.2)."""
    flight = synthetic_landing(t_td=205.0, on_ground_delay_samples=4.0)
    estimate = trained_estimator.estimate(flight)
    diag = estimate.diagnostics

    # Anchor identity, uncertainty, and diagnostics are all present.
    assert diag["physics_anchor_method"] == "decel_knee"
    assert math.isfinite(diag["physics_anchor_t_td"])
    assert math.isfinite(diag["physics_anchor_sigma_t"])
    assert diag["physics_anchor_sigma_t"] > 0.0
    assert diag["physics_anchor_confidence"] in ("normal", "low-confidence")
    assert isinstance(diag["physics_anchor_diagnostics"], dict)
    # The backbone's own diagnostics (e.g. the breakpoint) survive into the record.
    assert "breakpoint_time" in diag["physics_anchor_diagnostics"]

    # And the anchor t_td matches the standalone backbone estimate for this flight.
    standalone = DecelKneeEstimator().estimate(flight)
    assert diag["physics_anchor_t_td"] == pytest.approx(standalone.t_td, abs=1e-9)


@pytest.mark.property
@given(
    dt=st.floats(min_value=4.0, max_value=5.0),
    knee_t=st.floats(min_value=160.0, max_value=250.0),
    v_td_mps=st.floats(min_value=58.0, max_value=80.0),
)
def test_property_reconstruction_identity(trained_estimator, dt, knee_t, v_td_mps):
    """Feature: touchdown-point-detection, Task 17 hybrid residual.

    For any synthetic completed landing the (non-clamped) reported ``t_td``
    equals the physics anchor plus the predicted residual, and the physics
    anchor is always present in the record (Req 5.3, 6.2).
    """
    flight = synthetic_landing(
        dt=dt, t_td=knee_t, v_td_mps=v_td_mps, on_ground_delay_samples=4.0
    )
    estimate = trained_estimator.estimate(flight)
    diag = estimate.diagnostics
    assert "physics_anchor_t_td" in diag
    if estimate.confidence != "failed" and not diag.get("on_ground_clamped"):
        anchor_t_td = diag["physics_anchor_t_td"]
        residual = diag["predicted_residual_s"]
        assert estimate.t_td == pytest.approx(anchor_t_td + residual, abs=1e-9)


# ---------------------------------------------------------------------------
# Learning: corrects the physics-anchor bias
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trained_model_beats_anchor_baseline(labeled_dataset, trained_estimator):
    """Held-out ``t_td`` error beats the raw physics-anchor baseline (Req 5.3).

    The baseline is the physics anchor itself (predict residual = 0). The hybrid
    must reduce the held-out mean absolute error by learning the systematic bias
    injected between the knee and the QAR truth.
    """
    flights, truths = labeled_dataset
    backbone = DecelKneeEstimator()
    model_abs_err = []
    baseline_abs_err = []
    for flight, truth in zip(flights[50:], truths[50:]):
        estimate = trained_estimator.estimate(flight)
        anchor = backbone.estimate(flight)
        assert estimate.confidence in ("normal", "low-confidence")
        model_abs_err.append(abs(estimate.t_td - truth.touchdown_time_qar))
        baseline_abs_err.append(abs(anchor.t_td - truth.touchdown_time_qar))

    model_mae = float(np.mean(model_abs_err))
    baseline_mae = float(np.mean(baseline_abs_err))
    assert model_mae < 0.85 * baseline_mae, (
        f"model MAE {model_mae:.3f}s not better than anchor {baseline_mae:.3f}s"
    )


# ---------------------------------------------------------------------------
# Uncertainty from the residual quantile spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_quantile_spread_yields_positive_sigma_and_bracketing(
    labeled_dataset, trained_estimator
):
    """sigma_t is positive/finite, bounds are ordered, and the point is bracketed."""
    flights, _ = labeled_dataset
    for flight in flights[50:]:
        estimate = trained_estimator.estimate(flight)
        diag = estimate.diagnostics
        lo = diag["t_td_ci_lower"]
        hi = diag["t_td_ci_upper"]
        assert lo <= hi
        assert math.isfinite(estimate.sigma_t) and estimate.sigma_t > 0.0
        assert lo <= estimate.t_td <= hi


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_feature_importances_exposed_and_non_empty(trained_estimator):
    """Trained feature importances over the residual booster are exposed and non-trivial."""
    importances = trained_estimator.feature_importances()
    assert len(importances) > 0
    assert sum(importances.values()) > 0.0


# ---------------------------------------------------------------------------
# Unfit / failure behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_untrained_estimator_returns_failed_not_raises():
    """An untrained estimator returns a failed TDEstimate (does not raise)."""
    est = HybridResidualEstimator()
    assert est.is_trained is False
    estimate = est.estimate(synthetic_landing())
    assert estimate.confidence == "failed"
    assert math.isnan(estimate.t_td)
    assert estimate.diagnostics["detail"] == "model_not_trained"
    assert est.feature_importances() == {}


@pytest.mark.unit
def test_physics_anchor_failure_fails_gracefully(trained_estimator):
    """When the physics backbone fails, the hybrid fails but still records the anchor."""
    flight = synthetic_landing()
    flight.groundspeeds = np.full(flight.velocity_times.size, np.nan)  # backbone fails
    estimate = trained_estimator.estimate(flight)
    assert estimate.confidence == "failed"
    assert estimate.diagnostics["physics_anchor_method"] == "decel_knee"
    assert estimate.diagnostics["physics_anchor_confidence"] == "failed"
    assert estimate.diagnostics["reason_code"] == FailureReason.NO_GROUNDSPEED.value


# ---------------------------------------------------------------------------
# Inherited on-ground-flag upper bound (Requirement 18)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_ground_bound_enforced_via_base(trained_estimator):
    """A reconstructed t_td at/after the transition is clamped strictly below it."""
    flight = synthetic_landing(t_td=200.0, on_ground_delay_samples=-4.0)
    transition = flight.on_ground_transition_time
    estimate = trained_estimator.estimate(flight)
    assert estimate.confidence in ("normal", "low-confidence")
    assert estimate.t_td < transition
    assert estimate.diagnostics["on_ground_clamped"] is True
    assert "pre_clamp_t_td" in estimate.diagnostics


# ---------------------------------------------------------------------------
# Reproducibility (Req 15.1 / 15.2)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reproducible_same_seed_identical_predictions(labeled_dataset):
    """Training twice with the same seed yields bit-identical predictions."""
    flights, truths = labeled_dataset
    est_a = HybridResidualEstimator(n_estimators=120, num_leaves=15, seed=999)
    est_b = HybridResidualEstimator(n_estimators=120, num_leaves=15, seed=999)
    est_a.train(flights[:50], truths[:50])
    est_b.train(flights[:50], truths[:50])

    for flight in flights[50:]:
        ea = est_a.estimate(flight)
        eb = est_b.estimate(flight)
        assert ea.t_td == eb.t_td
        assert ea.sigma_t == eb.sigma_t
