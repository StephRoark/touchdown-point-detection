"""Tests for the grouped validation splits (Task 22.1).

Covers the primary tail-grouped three-way train/calibration/test partition
(Req 12.2, 12.4) and the separate held-out-airport / held-out-runway
generalization evaluations (Req 12.3), plus:

* **P10** -- Grouped Split No-Leakage: for any corpus, no test flight shares a
  tail with any train flight, the calibration partition is disjoint from both
  under the same grouping rule, and each held-out-airport (resp. -runway) test
  set contains no airport (resp. runway) present in its training set. The three
  groupings are evaluated separately, never intersected.
* determinism/reproducibility of the seeded splits (Req 15).
* unit tests for the group-preservation and fraction-allocation behaviour.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tdz.config.schema import (
    PipelineConfig,
    TDZConfig,
    ValidationConfig,
)
from tdz.models import QARTruthRecord
from tdz.validation import (
    FlightGroupKeys,
    make_validation_splits,
)
from tdz.validation.splits import _partition_counts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validation_config(
    *,
    primary_split_key: str = "tail",
    generalization_evals: list[str] | None = None,
    use_calibration_split: bool = True,
    train_fraction: float = 0.70,
    calibration_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> ValidationConfig:
    """A resolved ValidationConfig carrying the split knobs (other fields inert here)."""
    return ValidationConfig(
        primary_split_key=primary_split_key,
        generalization_evals=(
            ["airport", "runway"]
            if generalization_evals is None
            else generalization_evals
        ),
        use_calibration_split=use_calibration_split,
        train_fraction=train_fraction,
        calibration_fraction=calibration_fraction,
        test_fraction=test_fraction,
        min_stratum_size=30,
        cross_source=True,
        clock_offset_max_s=2.0,
        clock_drift_max_s=1.0,
        clock_xcorr_resample_dt_s=0.1,
        clock_max_lag_search_s=10.0,
        clock_min_overlap_s=20.0,
        clock_min_peak_correlation=0.5,
        clock_drift_segments=3,
        wrong_runway_lateral_margin_ft=50.0,
    )


def _tdz_config(
    validation: ValidationConfig, *, master_random_seed: int = 42
) -> TDZConfig:
    """A minimal TDZConfig sufficient to exercise the splitter.

    Only ``pipeline.master_random_seed`` and ``validation`` are consulted by the
    split builders; the remaining sections are populated with placeholders so the
    object is well-formed.
    """
    return TDZConfig(
        pipeline=PipelineConfig(master_random_seed=master_random_seed, ads_b_source="aireon"),
        timebase=None,  # type: ignore[arg-type]
        signals=None,  # type: ignore[arg-type]
        estimators=None,  # type: ignore[arg-type]
        fusion=None,  # type: ignore[arg-type]
        uncertainty=None,  # type: ignore[arg-type]
        quality_gates=None,  # type: ignore[arg-type]
        lever_arms=None,  # type: ignore[arg-type]
        geodesy=None,  # type: ignore[arg-type]
        vertical_crossing=None,  # type: ignore[arg-type]
        sources={},
        validation=validation,
        output=None,  # type: ignore[arg-type]
    )


def _tail_of(flight_id: str, flights: list[FlightGroupKeys]) -> str:
    return next(f.tail_number for f in flights if f.flight_id == flight_id)


def _airport_of(flight_id: str, flights: list[FlightGroupKeys]) -> str:
    return next(f.airport_id for f in flights if f.flight_id == flight_id)


def _runway_of(flight_id: str, flights: list[FlightGroupKeys]) -> str:
    return next(f.runway_id for f in flights if f.flight_id == flight_id)


# ---------------------------------------------------------------------------
# Hypothesis strategy: a corpus of flights with overlapping tail/airport/runway
# ---------------------------------------------------------------------------


@st.composite
def _flight_corpus(draw) -> list[FlightGroupKeys]:
    """Generate a corpus of flights whose grouping keys deliberately overlap.

    Tail numbers, airports and runways are each drawn from small pools so that
    many flights share each key (the interesting case for leakage control). Flight
    ids are unique. The three keys are drawn independently per flight, so the
    groupings genuinely differ from one another.
    """
    n = draw(st.integers(min_value=1, max_value=40))
    n_tails = draw(st.integers(min_value=1, max_value=12))
    n_airports = draw(st.integers(min_value=1, max_value=8))
    n_runways = draw(st.integers(min_value=1, max_value=10))

    tails = [f"N{i:03d}" for i in range(n_tails)]
    airports = [f"AP{i:02d}" for i in range(n_airports)]
    runways = [f"RW{i:02d}" for i in range(n_runways)]

    flights: list[FlightGroupKeys] = []
    for i in range(n):
        flights.append(
            FlightGroupKeys(
                flight_id=f"F{i:04d}",
                tail_number=draw(st.sampled_from(tails)),
                airport_id=draw(st.sampled_from(airports)),
                runway_id=draw(st.sampled_from(runways)),
            )
        )
    return flights


# ---------------------------------------------------------------------------
# Property 10: Grouped Split No-Leakage
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    flights=_flight_corpus(),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    use_calibration=st.booleans(),
)
def test_p10_grouped_split_no_leakage(flights, seed, use_calibration):
    """Feature: touchdown-point-detection, Property 10: Grouped Split No-Leakage

    For any primary train/test split, no test flight shares a tail number with any
    train flight, and the calibration partition is disjoint from both under the
    same grouping rule. For any held-out-airport (resp. -runway) evaluation, no
    test airport (resp. runway) appears in training. The three groupings are
    evaluated separately, not intersected.

    Validates: Requirements 12.2, 12.3, 12.4
    """
    config = _tdz_config(
        _validation_config(use_calibration_split=use_calibration),
        master_random_seed=seed,
    )
    splits = make_validation_splits(flights, config)

    # --- Every flight lands in exactly one primary partition (a partition) ----
    primary = splits.primary
    all_ids = {f.flight_id for f in flights}
    train_ids = set(primary.train)
    cal_ids = set(primary.calibration)
    test_ids = set(primary.test)
    assert train_ids.isdisjoint(cal_ids)
    assert train_ids.isdisjoint(test_ids)
    assert cal_ids.isdisjoint(test_ids)
    assert train_ids | cal_ids | test_ids == all_ids
    assert len(primary.train) + len(primary.calibration) + len(primary.test) == len(flights)

    # --- No tail leakage across the three primary partitions (Req 12.2, 12.4) -
    train_tails = {_tail_of(fid, flights) for fid in primary.train}
    cal_tails = {_tail_of(fid, flights) for fid in primary.calibration}
    test_tails = {_tail_of(fid, flights) for fid in primary.test}
    assert train_tails.isdisjoint(test_tails)
    assert cal_tails.isdisjoint(train_tails)
    assert cal_tails.isdisjoint(test_tails)

    if not use_calibration:
        assert primary.calibration == ()

    # --- Held-out-airport: no test airport present in training (Req 12.3) -----
    hoa = splits.held_out_airport
    assert hoa is not None
    train_airports = {_airport_of(fid, flights) for fid in hoa.train}
    test_airports = {_airport_of(fid, flights) for fid in hoa.test}
    assert train_airports.isdisjoint(test_airports)
    assert set(hoa.train).isdisjoint(set(hoa.test))
    assert set(hoa.train) | set(hoa.test) == all_ids

    # --- Held-out-runway: no test runway present in training (Req 12.3) -------
    hor = splits.held_out_runway
    assert hor is not None
    train_runways = {_runway_of(fid, flights) for fid in hor.train}
    test_runways = {_runway_of(fid, flights) for fid in hor.test}
    assert train_runways.isdisjoint(test_runways)
    assert set(hor.train).isdisjoint(set(hor.test))
    assert set(hor.train) | set(hor.test) == all_ids


# ---------------------------------------------------------------------------
# Determinism / reproducibility (Req 15)
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(flights=_flight_corpus(), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_splits_are_deterministic_in_seed(flights, seed):
    """Feature: touchdown-point-detection, Property 10 (determinism corollary)

    The same records, config and master seed reproduce identical partitions;
    input record order does not affect the result.

    Validates: Requirements 12.2, 12.3, 12.4
    """
    config = _tdz_config(_validation_config(), master_random_seed=seed)
    first = make_validation_splits(flights, config)
    second = make_validation_splits(list(reversed(flights)), config)

    # Compare as sets (input order must not matter), per partition.
    assert set(first.primary.train) == set(second.primary.train)
    assert set(first.primary.calibration) == set(second.primary.calibration)
    assert set(first.primary.test) == set(second.primary.test)
    assert set(first.held_out_airport.test) == set(second.held_out_airport.test)
    assert set(first.held_out_runway.test) == set(second.held_out_runway.test)


@pytest.mark.property
@given(flights=_flight_corpus(), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_different_seeds_stay_leakage_free(flights, seed):
    """A different master seed yields a valid (still leakage-free) split.

    Validates: Requirements 12.2
    """
    other_seed = (seed + 1) % (2**32)
    config = _tdz_config(_validation_config(), master_random_seed=other_seed)
    primary = make_validation_splits(flights, config).primary
    train_tails = {_tail_of(fid, flights) for fid in primary.train}
    test_tails = {_tail_of(fid, flights) for fid in primary.test}
    assert train_tails.isdisjoint(test_tails)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_whole_tail_group_stays_together():
    """All flights sharing a tail land in exactly one primary partition."""
    flights = [
        FlightGroupKeys(f"F{i}", tail_number=f"N{i % 3}", airport_id="AP0", runway_id="RW0")
        for i in range(30)
    ]
    config = _tdz_config(_validation_config(), master_random_seed=7)
    primary = make_validation_splits(flights, config).primary

    # Map each tail to the set of partitions it appears in.
    partitions = {"train": set(primary.train), "cal": set(primary.calibration), "test": set(primary.test)}
    by_tail: dict[str, set[str]] = {}
    for name, ids in partitions.items():
        for fid in ids:
            tail = next(f.tail_number for f in flights if f.flight_id == fid)
            by_tail.setdefault(tail, set()).add(name)
    for tail, parts in by_tail.items():
        assert len(parts) == 1, f"tail {tail} leaked across partitions {parts}"


@pytest.mark.unit
def test_calibration_disabled_folds_into_train():
    """With use_calibration_split=False the calibration partition is empty."""
    flights = [
        FlightGroupKeys(f"F{i}", tail_number=f"N{i}", airport_id="AP0", runway_id="RW0")
        for i in range(10)
    ]
    config = _tdz_config(
        _validation_config(use_calibration_split=False), master_random_seed=1
    )
    primary = make_validation_splits(flights, config).primary
    assert primary.calibration == ()
    assert len(primary.train) + len(primary.test) == 10


@pytest.mark.unit
def test_generalization_evals_optional():
    """Generalization splits are only built for keys listed in the config."""
    flights = [
        FlightGroupKeys(f"F{i}", tail_number=f"N{i}", airport_id=f"AP{i%2}", runway_id=f"RW{i%2}")
        for i in range(8)
    ]
    config = _tdz_config(
        _validation_config(generalization_evals=["airport"]), master_random_seed=3
    )
    splits = make_validation_splits(flights, config)
    assert splits.held_out_airport is not None
    assert splits.held_out_runway is None


@pytest.mark.unit
def test_accepts_qar_truth_records():
    """The splitter accepts raw QARTruthRecord objects, not only the key view."""
    records = [
        QARTruthRecord(
            flight_id=f"F{i}",
            touchdown_time_qar=0.0,
            touchdown_lat=0.0,
            touchdown_lon=0.0,
            clock_offset_estimate=None,
            clock_offset_quality="good",
            aircraft_type="B738",
            runway_id=f"RW{i%2}",
            airport_id=f"AP{i%2}",
            tail_number=f"N{i%3}",
        )
        for i in range(12)
    ]
    config = _tdz_config(_validation_config(), master_random_seed=5)
    splits = make_validation_splits(records, config)
    all_ids = {r.flight_id for r in records}
    got = set(splits.primary.train) | set(splits.primary.calibration) | set(splits.primary.test)
    assert got == all_ids


@pytest.mark.unit
def test_empty_corpus_yields_empty_splits():
    """An empty corpus produces empty partitions without error."""
    config = _tdz_config(_validation_config(), master_random_seed=0)
    splits = make_validation_splits([], config)
    assert splits.primary.train == ()
    assert splits.primary.calibration == ()
    assert splits.primary.test == ()
    assert splits.held_out_airport.train == ()
    assert splits.held_out_airport.test == ()


@pytest.mark.unit
@pytest.mark.parametrize(
    "n,fractions,expected_sum",
    [
        (10, [0.7, 0.15, 0.15], 10),
        (1, [0.7, 0.15, 0.15], 1),
        (0, [0.7, 0.15, 0.15], 0),
        (7, [1.0, 0.0], 7),
        (5, [0.0, 0.0, 0.0], 5),  # all-zero -> first bucket
    ],
)
def test_partition_counts_sum_preserved(n, fractions, expected_sum):
    """Largest-remainder allocation always sums to n and never goes negative."""
    counts = _partition_counts(n, fractions)
    assert sum(counts) == expected_sum
    assert all(c >= 0 for c in counts)
    assert len(counts) == len(fractions)


@pytest.mark.unit
def test_fraction_allocation_is_proportional():
    """Group-count allocation roughly matches the configured fractions."""
    # 100 distinct tails -> 100 groups -> counts equal group counts.
    flights = [
        FlightGroupKeys(f"F{i}", tail_number=f"N{i:03d}", airport_id="AP0", runway_id="RW0")
        for i in range(100)
    ]
    config = _tdz_config(
        _validation_config(train_fraction=0.6, calibration_fraction=0.2, test_fraction=0.2),
        master_random_seed=11,
    )
    primary = make_validation_splits(flights, config).primary
    assert len(primary.train) == 60
    assert len(primary.calibration) == 20
    assert len(primary.test) == 20
