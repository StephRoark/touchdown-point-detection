"""Tests for reproducibility & batch-level provenance (Task 23).

Covers Requirement 15 end-to-end:

* **Master-seed propagation** (Req 15.2): :func:`derive_component_seeds` fixes
  every stochastic component's stream from a single master seed, gives each an
  independent sub-seed, and is order-independent per name.
* **Batch provenance** (Req 15.3): :func:`resolve_batch_provenance` records the
  data version, git commit, model artifact hash, resolved-config hash, Python
  version, key-library versions, and the neural determinism mode, and projects
  cleanly down to the per-record :class:`~tdz.assemble.Provenance`.
* **Property 11 -- Deterministic Reproducibility** (Req 15.1/15.2): the
  deterministic pipeline (physics / change-point / geometry) produces
  bit-identical outputs across two runs on the same flight/config, over
  Hypothesis-randomized landings.
* **Two-source integration** (Req 15.1): identical underlying trajectories
  presented as Aireon (async velocity timebase) and FR24 (co-timed,
  velocity-only) each reproduce bit-identically run-to-run, with an identical
  batch provenance record.
* **LightGBM reproducibility** (Req 15.1): two boosters trained with the same
  master-seed-derived sub-seed give bit-identical predictions.

Thread-limit env vars (OMP/MKL/OPENBLAS = 1) are recommended when running this
module to keep the optional neural test from oversubscribing CPU threads.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.assemble import Provenance, compute_config_hash
from tdz.config.loader import load_config
from tdz.config.schema import TDZConfig
from tdz.models import QARTruthRecord
from tdz.pipeline import run_stage123
from tdz.reproducibility import (
    KEY_LIBRARIES,
    STOCHASTIC_COMPONENTS,
    BatchProvenance,
    compute_model_artifact_hash,
    derive_component_seeds,
    derive_seed,
    library_versions,
    neural_deterministic_mode,
    python_version,
    resolve_batch_provenance,
)
from tdz.tests.test_physics_estimators import synthetic_landing

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "tdz_config.yaml"


def _load_config() -> TDZConfig:
    return load_config(str(_CONFIG_PATH))


# Module-level config (property tests cannot take fixtures as arguments).
_CONFIG = _load_config()


@pytest.fixture(scope="module")
def config() -> TDZConfig:
    """The canonical repo configuration."""
    return _load_config()


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


def _bit_identical(a: float | None, b: float | None) -> bool:
    """Bit-identical comparison that treats NaN==NaN and None==None as equal."""
    if a is None or b is None:
        return a is None and b is None
    fa, fb = float(a), float(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    return fa == fb


# ===========================================================================
# Master-seed propagation (Req 15.2)
# ===========================================================================


@pytest.mark.unit
def test_derive_seeds_deterministic_and_independent():
    """Same master seed -> same sub-seeds; distinct components -> distinct seeds."""
    a = derive_component_seeds(42)
    b = derive_component_seeds(42)
    assert a == b  # deterministic in the master seed
    assert set(a.keys()) == set(STOCHASTIC_COMPONENTS)
    # Independent streams: the components do not collide onto one value.
    assert len(set(a.values())) == len(a)
    # Every sub-seed is a valid 32-bit unsigned integer (LightGBM range).
    assert all(0 <= s <= 2**32 - 1 for s in a.values())


@pytest.mark.unit
def test_derive_seeds_change_with_master_seed():
    """A different master seed yields different sub-seeds."""
    assert derive_component_seeds(42) != derive_component_seeds(43)


@pytest.mark.unit
def test_derive_seeds_order_independent_per_name():
    """A given (master_seed, name) maps to the same sub-seed regardless of the set."""
    full = derive_component_seeds(7, ["lightgbm", "sequence_model", "hybrid_residual"])
    subset = derive_component_seeds(7, ["hybrid_residual", "lightgbm"])
    assert subset["lightgbm"] == full["lightgbm"]
    assert subset["hybrid_residual"] == full["hybrid_residual"]
    # And the single-name helper agrees.
    assert derive_seed(7, "lightgbm") == full["lightgbm"]


# ===========================================================================
# Environment capture + batch provenance (Req 15.3)
# ===========================================================================


@pytest.mark.unit
def test_library_versions_complete_keyset():
    """Every key library appears; numpy is always present (a hard dependency)."""
    versions = library_versions()
    assert set(versions.keys()) == set(KEY_LIBRARIES)
    assert versions["numpy"] != "not-installed"
    assert versions["numpy"] == np.__version__


@pytest.mark.unit
def test_python_version_matches_runtime():
    import platform

    assert python_version() == platform.python_version()


@pytest.mark.unit
def test_resolve_batch_provenance_records_all_fields(config):
    """The batch provenance carries every Req-15.3 field and the config hash."""
    prov = resolve_batch_provenance(
        config,
        data_version="v2024.06",
        code_commit="deadbeef",
        model_artifact_hash="abc123",
    )
    assert isinstance(prov, BatchProvenance)
    assert prov.data_version == "v2024.06"
    assert prov.code_commit == "deadbeef"
    assert prov.model_artifact_hash == "abc123"
    assert prov.config_hash == compute_config_hash(config)
    assert prov.python_version == python_version()
    assert prov.library_versions["numpy"] == np.__version__
    # Mode defaults from config.pipeline.deterministic_mode.
    assert prov.neural_deterministic_mode == neural_deterministic_mode(config)


@pytest.mark.unit
def test_batch_provenance_records_configured_determinism_mode(config):
    """The neural determinism mode reflects config, and can be overridden."""
    config.pipeline.deterministic_mode = False
    prov = resolve_batch_provenance(config, code_commit="x")
    assert prov.neural_deterministic_mode is False
    forced = resolve_batch_provenance(
        config, code_commit="x", neural_deterministic_mode_used=True
    )
    assert forced.neural_deterministic_mode is True
    config.pipeline.deterministic_mode = True  # restore


@pytest.mark.unit
def test_batch_provenance_projects_to_record_provenance(config):
    """The batch record projects down to the per-record assembler provenance."""
    prov = resolve_batch_provenance(
        config, data_version="v1", code_commit="c1", model_artifact_hash="m1"
    )
    record = prov.to_record_provenance()
    assert isinstance(record, Provenance)
    assert record.data_version == "v1"
    assert record.code_commit == "c1"
    assert record.config_hash == prov.config_hash
    assert record.model_artifact_hash == "m1"


@pytest.mark.unit
def test_config_knob_default_and_hash(config):
    """The deterministic-mode knob defaults True and enters the resolved config hash."""
    assert config.pipeline.deterministic_mode is True
    assert config.resolved["pipeline"]["deterministic_mode"] is True


@pytest.mark.unit
def test_model_artifact_hash_stable(tmp_path):
    """The artifact hash is a stable SHA-256 of the file bytes."""
    import hashlib

    artifact = tmp_path / "model.bin"
    payload = b"trained-model-weights-payload"
    artifact.write_bytes(payload)
    got = compute_model_artifact_hash(str(artifact))
    assert got == hashlib.sha256(payload).hexdigest()
    # Re-hashing the same bytes is identical; different bytes differ.
    assert got == compute_model_artifact_hash(str(artifact))
    artifact.write_bytes(payload + b"!")
    assert compute_model_artifact_hash(str(artifact)) != got


# ===========================================================================
# Property 11: Deterministic Reproducibility
# ===========================================================================


@pytest.mark.property
@given(
    t_td=st.floats(min_value=150.0, max_value=260.0),
    dt=st.floats(min_value=4.0, max_value=5.0),
    v_td_mps=st.floats(min_value=58.0, max_value=80.0),
    approach_decel=st.floats(min_value=0.3, max_value=0.9),
    rollout_decel=st.floats(min_value=1.6, max_value=3.2),
    phase_s=st.floats(min_value=0.0, max_value=4.0),
    source=st.sampled_from(["aireon", "fr24"]),
)
def test_property_11_deterministic_reproducibility(
    t_td, dt, v_td_mps, approach_decel, rollout_decel, phase_s, source
):
    """Feature: touchdown-point-detection, Property 11

    Validates: Requirements 15.1, 15.2

    Running the deterministic pipeline (physics / change-point / geometry) twice
    with the same master seed, config and flight yields bit-identical numeric
    outputs for the reproducibility-guaranteed fields.
    """
    async_ts = source == "aireon"
    flight = synthetic_landing(
        t_td=t_td,
        dt=dt,
        v_td_mps=v_td_mps,
        approach_decel=approach_decel,
        rollout_decel=rollout_decel,
        phase_s=phase_s,
        velocity_offset_s=3.0 if async_ts else 0.0,
        omit_geometric=not async_ts,
        ads_b_source=source,
        flight_id="P11",
    )

    first = run_stage123(flight, _CONFIG)
    second = run_stage123(flight, _CONFIG)

    assert _bit_identical(first.combined_t_td, second.combined_t_td)
    assert _bit_identical(first.combined_distance_m, second.combined_distance_m)
    assert _bit_identical(first.lateral_offset_m, second.lateral_offset_m)
    assert first.combined.contributing == second.combined.contributing
    assert first.reason_code == second.reason_code
    # Every per-estimator numeric estimate is bit-identical too.
    assert set(first.estimates.keys()) == set(second.estimates.keys())
    for name, est in first.estimates.items():
        other = second.estimates[name]
        assert _bit_identical(est.t_td, other.t_td)
        assert _bit_identical(est.sigma_t, other.sigma_t)
        assert est.confidence == other.confidence


# ===========================================================================
# Two-source integration (Req 15.1): identical trajectory, both formats
# ===========================================================================


def _identical_trajectory_pair():
    """The same underlying landing presented as Aireon (async) and FR24 (co-timed).

    Aireon: geometric altitude present, velocity timebase offset from the
    position timebase (async timestamps). FR24: velocity-only (no geometric
    altitude), velocity co-timed with position. The underlying dynamics
    (touchdown time, speeds, decelerations) are identical.
    """
    common = dict(
        t_td=205.0,
        dt=4.5,
        v_td_mps=68.0,
        approach_decel=0.5,
        rollout_decel=2.5,
    )
    aireon = synthetic_landing(
        **common,
        velocity_offset_s=3.0,
        omit_geometric=False,
        ads_b_source="aireon",
        flight_id="TRAJ",
    )
    fr24 = synthetic_landing(
        **common,
        velocity_offset_s=0.0,
        omit_geometric=True,
        ads_b_source="fr24",
        flight_id="TRAJ",
    )
    return aireon, fr24


def _assert_run_bit_identical(config, flight):
    first = run_stage123(flight, config)
    second = run_stage123(flight, config)
    assert _bit_identical(first.combined_t_td, second.combined_t_td)
    assert _bit_identical(first.combined_distance_m, second.combined_distance_m)
    assert _bit_identical(first.lateral_offset_m, second.lateral_offset_m)
    assert first.combined.contributing == second.combined.contributing
    return first


@pytest.mark.integration
def test_reproducibility_aireon_async_source(config):
    """Aireon async-timestamp trajectory reproduces bit-identically (Req 15.1)."""
    aireon, _ = _identical_trajectory_pair()
    result = _assert_run_bit_identical(config, aireon)
    assert result.combined.ok  # a real touchdown was produced


@pytest.mark.integration
def test_reproducibility_fr24_cotimed_source(config):
    """FR24 co-timed velocity-only trajectory reproduces bit-identically (Req 15.1)."""
    _, fr24 = _identical_trajectory_pair()
    result = _assert_run_bit_identical(config, fr24)
    assert result.combined.ok


@pytest.mark.integration
def test_reproducibility_batch_provenance_identical_across_runs(config):
    """The batch provenance record is identical across repeated runs (Req 15.3)."""
    prov1 = resolve_batch_provenance(
        config, data_version="v1", code_commit="fixed-commit"
    )
    prov2 = resolve_batch_provenance(
        config, data_version="v1", code_commit="fixed-commit"
    )
    assert prov1 == prov2
    assert prov1.config_hash == compute_config_hash(config)
    assert prov1.python_version == python_version()
    assert prov1.library_versions == library_versions()


# ===========================================================================
# LightGBM reproducibility from the master seed (Req 15.1)
# ===========================================================================


@pytest.mark.integration
def test_lightgbm_reproducible_from_master_seed():
    """Two LightGBM boosters seeded from the same master seed predict identically."""
    lightgbm = pytest.importorskip("lightgbm")  # noqa: F841
    from tdz.estimators.learned import LightGbmTouchdownEstimator

    rng = np.random.default_rng(20240601)
    flights, truths = [], []
    for i in range(40):
        t_td = 200.0 + float(rng.uniform(-40.0, 40.0))
        flights.append(
            synthetic_landing(
                dt=float(rng.uniform(4.0, 5.0)),
                t_td=t_td,
                v_td_mps=float(rng.uniform(58.0, 80.0)),
                approach_decel=float(rng.uniform(0.3, 0.9)),
                rollout_decel=float(rng.uniform(1.6, 3.2)),
                flight_id=f"F{i:03d}",
            )
        )
        truths.append(_truth(f"F{i:03d}", t_td))

    # Seed both estimators from the SAME master-seed-derived sub-seed (Req 15.2).
    seed = derive_seed(_CONFIG.pipeline.master_random_seed, "lightgbm")

    est_a = LightGbmTouchdownEstimator(n_estimators=120, num_leaves=15, seed=seed)
    est_a.train(flights[:30], truths[:30])
    est_b = LightGbmTouchdownEstimator(n_estimators=120, num_leaves=15, seed=seed)
    est_b.train(flights[:30], truths[:30])

    for flight in flights[30:]:
        ea = est_a.estimate(flight)
        eb = est_b.estimate(flight)
        assert _bit_identical(ea.t_td, eb.t_td)
        assert _bit_identical(ea.sigma_t, eb.sigma_t)
        assert ea.confidence == eb.confidence


# ===========================================================================
# Neural deterministic mode (Req 15.2): bit-identical in deterministic mode
# ===========================================================================


@pytest.mark.integration
@pytest.mark.slow
def test_sequence_model_bit_identical_in_deterministic_mode():
    """The neural model reproduces bit-identically in the explicit deterministic mode."""
    pytest.importorskip("torch")
    from tdz.estimators.learned import SequenceModelEstimator

    rng = np.random.default_rng(99)
    flights, truths = [], []
    for i in range(10):
        t_td = 200.0 + float(rng.uniform(-20.0, 20.0))
        flights.append(
            synthetic_landing(
                dt=float(rng.uniform(4.0, 5.0)),
                t_td=t_td,
                v_td_mps=float(rng.uniform(60.0, 78.0)),
                flight_id=f"S{i:02d}",
            )
        )
        truths.append(_truth(f"S{i:02d}", t_td))

    seed = derive_seed(_CONFIG.pipeline.master_random_seed, "sequence_model")

    def _train():
        est = SequenceModelEstimator(
            n_epochs=15, seed=seed, deterministic=True, hidden_dim=16
        )
        est.train(flights[:8], truths[:8])
        return est

    est_a = _train()
    est_b = _train()
    for flight in flights[8:]:
        ea = est_a.estimate(flight)
        eb = est_b.estimate(flight)
        assert ea.diagnostics.get("deterministic_mode") is True
        assert _bit_identical(ea.t_td, eb.t_td)
        assert _bit_identical(ea.sigma_t, eb.sigma_t)
