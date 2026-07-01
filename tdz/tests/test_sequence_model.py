"""Tests for the TCN/BiLSTM sequence model and rare-type fallback (Task 16).

Two independent concerns are exercised here:

* **Property 15 -- Rare-Type Physics Fallback** (Req 6.3/6.4): a Hypothesis
  property over flights whose aircraft type has fewer than the configured
  threshold of training flights, asserting the physics estimator is the primary
  contributor, the learned estimate is omitted, and the physics anchor is always
  present. This path never invokes the learned model, so it runs **without
  PyTorch** -- the property is about wiring, not the network.

* **The sequence model itself** (Req 5.3/6.2): sequence assembly, soft Gaussian
  labels, training, the per-timestep distribution -> expected-value ``t_td`` +
  distribution-width uncertainty, the optional deep ensemble, the inherited
  on-ground bound, and reproducibility. These require PyTorch and are skipped
  gracefully (via ``importorskip``) when it is unavailable.

Tolerances stay honest to the 4-5 s ADS-B cadence: the network is small and
trained briefly on synthetic data, so accuracy is asserted against the trivial
physics-knee baseline rather than an unrealistic sub-second target.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tdz.estimators.learned import (
    LEARNED_PRIMARY,
    PHYSICS_PRIMARY,
    RareTypePhysicsFallback,
    SequenceModelEstimator,
    build_sequence_input,
    training_flight_counts,
)
from tdz.estimators.learned.sequence_model import (
    MIN_SEQUENCE_SAMPLES,
    N_SOURCE_CODES,
    _soft_gaussian_label,
)
from tdz.estimators.physics import DecelKneeEstimator
from tdz.models import FailureReason, QARTruthRecord
from tdz.tests.test_physics_estimators import synthetic_landing


# ---------------------------------------------------------------------------
# Shared synthetic-dataset helpers
# ---------------------------------------------------------------------------


def _truth(flight_id: str, t_td: float, aircraft_type: str = "B738") -> QARTruthRecord:
    """A minimal QAR truth record carrying the known touchdown time and type."""
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


def _make_dataset(n: int, seed: int, aircraft_type: str = "B738"):
    """``n`` varied synthetic labeled landings of one aircraft type."""
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
            flight_id=f"{aircraft_type}{i:03d}",
        )
        flight.aircraft_type = aircraft_type
        flights.append(flight)
        truths.append(_truth(f"{aircraft_type}{i:03d}", t_td, aircraft_type))
    return flights, truths


# ===========================================================================
# Sequence assembly + soft labels (torch-free)
# ===========================================================================


@pytest.mark.unit
def test_sequence_input_assembles_on_velocity_timebase():
    """A completed landing yields aligned channels, flags, and a finite reference."""
    flight = synthetic_landing(t_td=205.0)
    seq, reason = build_sequence_input(flight)
    assert reason is None and seq is not None
    t = seq.times.size
    assert t >= MIN_SEQUENCE_SAMPLES
    assert seq.continuous.shape == (t, 8)
    assert seq.flags.shape == (t, 2)
    assert math.isfinite(seq.reference_time)
    # Geometric altitude is present -> height channel available.
    assert seq.flags[0, 1] == 1.0
    assert 0 <= seq.source_index < N_SOURCE_CODES


@pytest.mark.unit
def test_sequence_input_no_groundspeed_fails():
    """No groundspeed at all -> NO_GROUNDSPEED, no sequence."""
    flight = synthetic_landing()
    flight.groundspeeds = np.full(flight.velocity_times.size, np.nan)
    seq, reason = build_sequence_input(flight)
    assert seq is None
    assert reason == FailureReason.NO_GROUNDSPEED


@pytest.mark.unit
def test_sequence_input_too_few_samples_fails():
    """Too few groundspeed samples -> INSUFFICIENT_SAMPLES."""
    flight = synthetic_landing(n_before=1, n_after=1)
    seq, reason = build_sequence_input(flight)
    assert seq is None
    assert reason == FailureReason.INSUFFICIENT_SAMPLES


@pytest.mark.unit
def test_velocity_only_source_height_unavailable():
    """An FR24-like velocity-only source has no height channel (flag = 0)."""
    flight = synthetic_landing(omit_geometric=True)
    seq, reason = build_sequence_input(flight)
    assert reason is None and seq is not None
    assert seq.flags[0, 1] == 0.0  # height unavailable
    # Distance still derives from lat/lon, so it stays available.
    assert seq.flags[0, 0] == 1.0


@pytest.mark.unit
def test_soft_gaussian_label_is_a_normalised_bump():
    """The soft label sums to 1 and peaks at the sample nearest touchdown."""
    times = np.array([190.0, 195.0, 200.0, 205.0, 210.0])
    label = _soft_gaussian_label(times, t_td=201.0, label_sigma_s=2.5)
    assert label.shape == times.shape
    assert math.isclose(float(label.sum()), 1.0, rel_tol=1e-9)
    assert int(np.argmax(label)) == 2  # nearest to 201.0 is 200.0
    assert np.all(label >= 0.0)


@pytest.mark.unit
def test_soft_gaussian_label_far_touchdown_falls_back_to_nearest():
    """When the bump underflows everywhere, mass lands on the single nearest sample."""
    times = np.array([100.0, 105.0, 110.0])
    label = _soft_gaussian_label(times, t_td=1.0e6, label_sigma_s=2.0)
    assert math.isclose(float(label.sum()), 1.0, rel_tol=1e-9)
    assert int(np.argmax(label)) == 2  # nearest (largest) time


# ===========================================================================
# Property 15: Rare-Type Physics Fallback (torch-free)
# ===========================================================================


@pytest.mark.unit
def test_training_flight_counts_counts_per_type():
    """Counts are per aircraft type regardless of source (Req 6.3)."""
    truths = [
        _truth("a", 200.0, "B738"),
        _truth("b", 201.0, "B738"),
        _truth("c", 202.0, "A320"),
    ]
    counts = training_flight_counts(truths)
    assert counts == {"B738": 2, "A320": 1}


@pytest.mark.unit
def test_rare_type_uses_physics_primary_and_omits_learned():
    """A rare type -> physics primary, learned omitted, anchor present (Req 6.3)."""
    flight = synthetic_landing(t_td=200.0)
    flight.aircraft_type = "RARE1"
    selector = RareTypePhysicsFallback(
        DecelKneeEstimator(),
        SequenceModelEstimator(),     # never invoked on the rare path (no torch needed)
        training_type_counts={"RARE1": 3},
        threshold=50,
    )
    result = selector.select(flight)
    assert result.is_rare_type is True
    assert result.primary_source == PHYSICS_PRIMARY
    assert result.learned_estimate is None
    assert result.physics_anchor is not None
    assert result.physics_anchor.method_name == "decel_knee"
    assert not result.touchdown_omitted
    assert math.isfinite(result.t_td)


@pytest.mark.unit
def test_rare_type_physics_failure_omits_touchdown_not_learned_fallback():
    """Rare type + physics failure -> low-confidence, no touchdown, no learned fallback (Req 6.4)."""
    flight = synthetic_landing()
    flight.aircraft_type = "RARE2"
    flight.groundspeeds = np.full(flight.velocity_times.size, np.nan)  # physics fails
    selector = RareTypePhysicsFallback(
        DecelKneeEstimator(),
        SequenceModelEstimator(),
        training_type_counts={"RARE2": 1},
        threshold=50,
    )
    result = selector.select(flight)
    assert result.is_rare_type is True
    assert result.primary_source == PHYSICS_PRIMARY
    assert result.learned_estimate is None        # never fell back to learned
    assert result.touchdown_omitted is True
    assert result.confidence == "low-confidence"
    assert result.reason_code is not None
    assert math.isnan(result.t_td)
    assert result.physics_anchor is not None       # anchor still in the record


@pytest.mark.property
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    n_flights=st.integers(min_value=0, max_value=49),
    threshold=st.integers(min_value=50, max_value=60),
    t_td=st.floats(min_value=160.0, max_value=240.0),
    dt=st.floats(min_value=4.0, max_value=5.0),
)
def test_property_15_rare_type_physics_fallback(n_flights, threshold, t_td, dt):
    """Property 15 (Req 6.3): touchdown-point-detection rare-type physics fallback.

    For ANY aircraft type with fewer than the threshold training flights, the
    physics estimator is the primary contributor, the learned estimate is
    omitted, and the physics anchor is present in the record.

    **Validates: Requirements 6.3**
    """
    flight = synthetic_landing(t_td=t_td, dt=dt)
    flight.aircraft_type = "RAREPROP"
    selector = RareTypePhysicsFallback(
        DecelKneeEstimator(),
        SequenceModelEstimator(),
        training_type_counts={"RAREPROP": n_flights},
        threshold=threshold,
    )
    result = selector.select(flight)

    assert result.is_rare_type is True
    assert result.primary_source == PHYSICS_PRIMARY
    assert result.learned_estimate is None
    assert result.physics_anchor is not None
    # If a touchdown is produced it must come from physics (anchor) and be bounded.
    if not result.touchdown_omitted:
        assert math.isfinite(result.t_td)
        if flight.on_ground_transition_time is not None:
            assert result.t_td < flight.on_ground_transition_time


# ===========================================================================
# Sequence model: training, distribution -> t_td + width, ensemble (torch)
# ===========================================================================

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def labeled_dataset():
    """A fixed synthetic dataset: 44 B738 landings, split 34 train / 10 held-out."""
    return _make_dataset(44, seed=20240602)


@pytest.fixture(scope="module")
def trained_estimator(labeled_dataset):
    """A single-model sequence estimator trained on the first 34 landings."""
    flights, truths = labeled_dataset
    est = SequenceModelEstimator(
        hidden_dim=24, n_epochs=80, learning_rate=0.03, seed=7, n_ensemble=1
    )
    est.train(flights[:34], truths[:34])
    return est


@pytest.mark.unit
def test_untrained_estimator_returns_failed_not_raises():
    """An untrained sequence estimator returns a failed estimate (does not raise)."""
    est = SequenceModelEstimator()
    assert est.is_trained is False
    estimate = est.estimate(synthetic_landing())
    assert estimate.confidence == "failed"
    assert math.isnan(estimate.t_td)
    assert estimate.diagnostics["detail"] == "model_not_trained"


@pytest.mark.unit
def test_train_records_per_type_counts(labeled_dataset, trained_estimator):
    """Training records per-type counts for the rare-type fallback."""
    assert trained_estimator.training_type_counts.get("B738") == 34


@pytest.mark.unit
def test_distribution_yields_expected_value_and_width(trained_estimator, labeled_dataset):
    """The per-timestep distribution gives a finite t_td and positive width sigma."""
    flights, _ = labeled_dataset
    mean_p, seq, members = trained_estimator.predict_distribution(flights[34])
    assert mean_p is not None and seq is not None
    assert math.isclose(float(mean_p.sum()), 1.0, rel_tol=1e-5)
    estimate = trained_estimator.estimate(flights[34])
    assert estimate.confidence in ("normal", "low-confidence")
    assert math.isfinite(estimate.t_td)
    assert estimate.sigma_t > 0.0
    assert estimate.diagnostics["distribution_width_s"] >= 0.0
    assert estimate.diagnostics["n_timesteps"] == seq.times.size


@pytest.mark.unit
def test_trained_model_beats_breakpoint_baseline(labeled_dataset, trained_estimator):
    """Held-out t_td error beats the trivial physics-breakpoint baseline (Req 5.3)."""
    flights, truths = labeled_dataset
    model_err = []
    baseline_err = []
    for flight, truth in zip(flights[34:], truths[34:]):
        estimate = trained_estimator.estimate(flight)
        _, seq, _ = trained_estimator.predict_distribution(flight)
        model_err.append(abs(estimate.t_td - truth.touchdown_time_qar))
        baseline_err.append(abs(seq.reference_time - truth.touchdown_time_qar))
    model_mae = float(np.mean(model_err))
    baseline_mae = float(np.mean(baseline_err))
    assert model_mae <= baseline_mae, (
        f"model MAE {model_mae:.3f}s worse than baseline {baseline_mae:.3f}s"
    )


@pytest.mark.unit
def test_deep_ensemble_reports_epistemic_uncertainty(labeled_dataset):
    """A deep ensemble reports a non-negative epistemic term in diagnostics."""
    flights, truths = labeled_dataset
    est = SequenceModelEstimator(
        hidden_dim=16, n_epochs=40, learning_rate=0.05, seed=11, n_ensemble=3
    )
    est.train(flights[:34], truths[:34])
    estimate = est.estimate(flights[34])
    assert estimate.diagnostics["n_ensemble"] == 3
    assert estimate.diagnostics["epistemic_sigma_s"] >= 0.0
    assert math.isfinite(estimate.t_td)


@pytest.mark.unit
def test_on_ground_bound_enforced_via_base(trained_estimator):
    """A t_td at/after the on-ground transition is clamped strictly below it (Req 18)."""
    flight = synthetic_landing(t_td=200.0, on_ground_delay_samples=-4.0)
    transition = flight.on_ground_transition_time
    estimate = trained_estimator.estimate(flight)
    assert estimate.t_td < transition
    assert estimate.diagnostics["on_ground_clamped"] is True


@pytest.mark.unit
def test_reproducible_same_seed_identical_predictions(labeled_dataset):
    """Training twice with the same seed yields identical CPU predictions (Req 15.1/15.2)."""
    flights, truths = labeled_dataset
    est_a = SequenceModelEstimator(hidden_dim=16, n_epochs=40, seed=123, n_ensemble=1)
    est_b = SequenceModelEstimator(hidden_dim=16, n_epochs=40, seed=123, n_ensemble=1)
    est_a.train(flights[:34], truths[:34])
    est_b.train(flights[:34], truths[:34])
    for flight in flights[34:]:
        ea = est_a.estimate(flight)
        eb = est_b.estimate(flight)
        assert ea.t_td == eb.t_td
        assert ea.sigma_t == eb.sigma_t


@pytest.mark.unit
def test_common_type_uses_learned_primary_with_anchor(labeled_dataset, trained_estimator):
    """A common type -> learned primary with the physics anchor still present (Req 6.2)."""
    flights, _ = labeled_dataset
    selector = RareTypePhysicsFallback(
        DecelKneeEstimator(),
        trained_estimator,
        training_type_counts={"B738": 34},
        threshold=10,
    )
    result = selector.select(flights[34])
    assert result.is_rare_type is False
    assert result.primary_source == LEARNED_PRIMARY
    assert result.learned_estimate is not None
    assert result.physics_anchor is not None          # anchor always included
    assert result.primary_method == "sequence_model"
