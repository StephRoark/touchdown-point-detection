"""Tests for the LightGBM window-feature estimator (Task 15).

The first learned estimator (Req 5.3): per-landing engineered features ->
gradient-boosted trees predicting a touchdown-time **offset** relative to the
physics-knee reference, with a quantile pair for uncertainty (design "Learned
Estimators Detail -> LightGBM Window Features").

Coverage
--------
* **Feature extraction** -- fixed length, no truth leakage, and graceful
  failure (no groundspeed / too few samples). A Hypothesis property asserts the
  vector is always fixed-length with a finite offset reference.
* **Learning** -- after training on a small synthetic set the held-out
  touchdown error beats the trivial physics-breakpoint baseline (the model
  refines physics, Req 5.3 / design principle 2).
* **Uncertainty** -- the quantile spread yields a positive finite ``sigma_t``,
  ordered reconstructed bounds, and the median lies within those bounds for the
  large majority of held-out flights.
* **Interpretability** -- feature importances are exposed and non-empty.
* **Unfit behaviour** -- an untrained estimator returns a *failed* estimate
  (it does not raise).
* **On-ground bound** -- the inherited Requirement-18 upper bound still clamps a
  reconstructed ``t_td`` that lands at/after the on-ground transition.
* **Reproducibility** -- training twice with the same seed gives identical
  predictions (Req 15.1/15.2; deterministic CPU settings, ``num_threads=1``).

LightGBM is imported via ``importorskip`` so the suite degrades gracefully if it
is unavailable; it is a declared dependency and expected to be installed.

Tolerances are kept honest to the 4-5 s ADS-B cadence: the synthetic landings
vary touchdown time, speeds, deceleration and cadence so the model has signal,
but absolute errors are reported against the cadence-limited baseline rather
than an unrealistic sub-second target.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.estimators.learned import (
    FEATURE_NAMES,
    N_FEATURES,
    LightGbmTouchdownEstimator,
    extract_window_features,
)
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

    Touchdown time, approach speed, approach/rollout deceleration and sample
    cadence are all varied so the engineered features carry real signal about
    where ``t_td`` sits relative to the physics-knee reference.
    """
    rng = np.random.default_rng(seed)
    flights = []
    truths = []
    for i in range(n):
        t_td = 200.0 + float(rng.uniform(-40.0, 40.0))
        flight = synthetic_landing(
            dt=float(rng.uniform(4.0, 5.0)),
            t_td=t_td,
            n_before=int(rng.integers(10, 16)),
            n_after=int(rng.integers(6, 11)),
            v_td_mps=float(rng.uniform(58.0, 80.0)),
            approach_decel=float(rng.uniform(0.3, 0.9)),
            rollout_decel=float(rng.uniform(1.6, 3.2)),
            on_ground_delay_samples=float(rng.uniform(1.5, 3.0)),
            flight_id=f"F{i:03d}",
        )
        flights.append(flight)
        truths.append(_truth(f"F{i:03d}", t_td))
    return flights, truths


@pytest.fixture(scope="module")
def labeled_dataset():
    """A fixed synthetic dataset: 64 landings, split 50 train / 14 held-out."""
    flights, truths = _make_dataset(64, seed=20240601)
    return flights, truths


@pytest.fixture(scope="module")
def trained_estimator(labeled_dataset):
    """A LightGBM estimator trained on the first 50 landings (fast settings)."""
    flights, truths = labeled_dataset
    est = LightGbmTouchdownEstimator(n_estimators=150, num_leaves=15, seed=12345)
    est.train(flights[:50], truths[:50])
    return est


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_features_fixed_length_and_named():
    """The feature vector is fixed length and matches the documented schema."""
    flight = synthetic_landing()
    features, reason = extract_window_features(flight)
    assert reason is None
    assert features is not None
    assert features.values.shape == (N_FEATURES,)
    assert features.names == FEATURE_NAMES
    assert math.isfinite(features.reference_time)
    assert features.reference_kind == "segmented_breakpoint"


@pytest.mark.unit
def test_features_no_truth_leakage():
    """Features depend only on the flight record, never on QAR truth.

    Extraction takes no truth argument; re-extracting yields the identical
    vector, confirming the features cannot encode the label.
    """
    flight = synthetic_landing(t_td=212.3)
    f1, _ = extract_window_features(flight)
    f2, _ = extract_window_features(flight)
    assert np.array_equal(f1.values, f2.values, equal_nan=True)


@pytest.mark.unit
def test_features_no_groundspeed_fails():
    """No groundspeed at all -> NO_GROUNDSPEED, no feature vector."""
    flight = synthetic_landing()
    flight.groundspeeds = np.full(flight.velocity_times.size, np.nan)
    features, reason = extract_window_features(flight)
    assert features is None
    assert reason == FailureReason.NO_GROUNDSPEED


@pytest.mark.unit
def test_features_too_few_samples_fails():
    """Too few groundspeed samples to fit the reference -> INSUFFICIENT_SAMPLES."""
    flight = synthetic_landing(n_before=1, n_after=1)
    features, reason = extract_window_features(flight)
    assert features is None
    assert reason == FailureReason.INSUFFICIENT_SAMPLES


@pytest.mark.property
@given(
    dt=st.floats(min_value=4.0, max_value=5.0),
    t_td=st.floats(min_value=150.0, max_value=260.0),
    v_td_mps=st.floats(min_value=55.0, max_value=82.0),
    approach_decel=st.floats(min_value=0.3, max_value=0.9),
    rollout_decel=st.floats(min_value=1.6, max_value=3.4),
    omit_geometric=st.booleans(),
)
def test_property_feature_vector_fixed_length(
    dt, t_td, v_td_mps, approach_decel, rollout_decel, omit_geometric
):
    """Feature: touchdown-point-detection, Task 15 LightGBM window features.

    For any synthetic completed landing the extractor returns a fixed-length
    vector (``N_FEATURES``) with a finite offset reference; unavailable features
    may be NaN but the length and reference are always well-defined.
    """
    flight = synthetic_landing(
        dt=dt,
        t_td=t_td,
        v_td_mps=v_td_mps,
        approach_decel=approach_decel,
        rollout_decel=rollout_decel,
        omit_geometric=omit_geometric,
    )
    features, reason = extract_window_features(flight)
    assert reason is None
    assert features.values.shape == (N_FEATURES,)
    assert math.isfinite(features.reference_time)


# ---------------------------------------------------------------------------
# Learning: beats the trivial baseline
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trained_model_beats_breakpoint_baseline(labeled_dataset, trained_estimator):
    """Held-out ``t_td`` error beats the trivial physics-breakpoint baseline.

    The baseline is the offset reference itself (predict offset = 0, i.e.
    ``t_td = breakpoint``). A model that has learned anything must reduce the
    held-out mean absolute error relative to that baseline (Req 5.3).
    """
    flights, truths = labeled_dataset
    model_abs_err = []
    baseline_abs_err = []
    for flight, truth in zip(flights[50:], truths[50:]):
        estimate = trained_estimator.estimate(flight)
        features, _ = extract_window_features(flight)
        assert estimate.confidence in ("normal", "low-confidence")
        model_abs_err.append(abs(estimate.t_td - truth.touchdown_time_qar))
        baseline_abs_err.append(abs(features.reference_time - truth.touchdown_time_qar))

    model_mae = float(np.mean(model_abs_err))
    baseline_mae = float(np.mean(baseline_abs_err))
    # The model must be a clear improvement, not a coin-flip tie.
    assert model_mae < 0.85 * baseline_mae, (
        f"model MAE {model_mae:.3f}s not better than baseline {baseline_mae:.3f}s"
    )


# ---------------------------------------------------------------------------
# Uncertainty from the quantile spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_quantile_spread_yields_positive_sigma_and_bracketing(
    labeled_dataset, trained_estimator
):
    """sigma_t is positive/finite, bounds are ordered, and the median is bracketed.

    Quantile rearrangement guarantees ``q_low <= median <= q_high`` even when the
    independently-fit boosters would otherwise cross, so the median lies within
    the reconstructed bounds for every held-out flight, with a positive finite
    sigma throughout.
    """
    flights, truths = labeled_dataset
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
    """Trained feature importances are exposed, named, and non-trivial."""
    importances = trained_estimator.feature_importances()
    assert set(importances.keys()) == set(FEATURE_NAMES)
    assert len(importances) == N_FEATURES
    # At least one feature carries positive importance (the model split on data).
    assert sum(importances.values()) > 0.0


# ---------------------------------------------------------------------------
# Unfit behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_untrained_estimator_returns_failed_not_raises():
    """An untrained estimator returns a failed TDEstimate (does not raise)."""
    est = LightGbmTouchdownEstimator()
    assert est.is_trained is False
    flight = synthetic_landing()
    estimate = est.estimate(flight)
    assert estimate.confidence == "failed"
    assert math.isnan(estimate.t_td)
    assert estimate.diagnostics["detail"] == "model_not_trained"
    # Importances degrade to empty when untrained (no booster to query).
    assert est.feature_importances() == {}


@pytest.mark.unit
def test_trained_estimator_no_groundspeed_fails(trained_estimator):
    """Even when trained, a flight with no groundspeed fails gracefully."""
    flight = synthetic_landing()
    flight.groundspeeds = np.full(flight.velocity_times.size, np.nan)
    estimate = trained_estimator.estimate(flight)
    assert estimate.confidence == "failed"
    assert estimate.diagnostics["reason_code"] == FailureReason.NO_GROUNDSPEED.value


# ---------------------------------------------------------------------------
# Inherited on-ground-flag upper bound (Requirement 18)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_on_ground_bound_enforced_via_base(trained_estimator):
    """A reconstructed t_td at/after the transition is clamped strictly below it.

    The on-ground flag is forced to transition well BEFORE the true knee, so the
    learned ``t_td`` (~the physics breakpoint) lands after the bound and the base
    :class:`PhysicsEstimator` clamps it (Req 18.1-18.3; Property 5).
    """
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
    """Training twice with the same seed yields bit-identical predictions.

    LightGBM is deterministic on CPU with a fixed seed and single thread
    (``deterministic=True``, ``force_row_wise=True``, ``num_threads=1``), and the
    categorical encodings use a fixed deterministic hash, so two independently
    trained estimators produce identical outputs (Req 15.1, 15.2).
    """
    flights, truths = labeled_dataset
    est_a = LightGbmTouchdownEstimator(n_estimators=120, num_leaves=15, seed=999)
    est_b = LightGbmTouchdownEstimator(n_estimators=120, num_leaves=15, seed=999)
    est_a.train(flights[:50], truths[:50])
    est_b.train(flights[:50], truths[:50])

    for flight in flights[50:]:
        ea = est_a.estimate(flight)
        eb = est_b.estimate(flight)
        assert ea.t_td == eb.t_td
        assert ea.sigma_t == eb.sigma_t
