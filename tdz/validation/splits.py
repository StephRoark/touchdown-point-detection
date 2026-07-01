"""Grouped train/calibration/test splits for the validation harness (Task 22.1).

This module builds the leakage-controlled partitions the validation harness
evaluates against (Req 12.2, 12.3, 12.4; Property 10):

* **Primary tail-grouped three-way split** (train / calibration / test). Every
  flight sharing an aircraft *tail number* (more generally, the configured
  ``primary_split_key``) is assigned to exactly one partition, so no flight in
  the test partition shares a tail with any flight in train, and the calibration
  partition is disjoint from both under the same grouping rule (Req 12.2, 12.4).
  Tail-grouping is the leakage control that stops a model memorizing an
  individual airframe's sensor biases; the calibration partition is reserved for
  conformalizing the uncertainty intervals so reported coverage is measured on
  data unseen during both model fitting and interval calibration.

* **Separate generalization-stress evaluations** (Req 12.3): a *held-out-airport*
  evaluation (test airports absent from training) and a *held-out-runway*
  evaluation (test runways absent from training). These are produced and reported
  **alongside** the primary tail-grouped metrics -- they are **not** intersected
  into one split, since intersecting all three groupings starves data and
  conflates leakage control with geographic generalization.

Determinism (Req 15)
--------------------
Every partition is driven by ``config.pipeline.master_random_seed`` via a
:class:`numpy.random.SeedSequence`, whose independent spawned children seed the
primary split and each generalization split. Running the splitter twice with the
same records, config and seed yields identical partitions. Group keys are sorted
before shuffling so the result is independent of input record order. All split
fractions live in config (``validation.train_fraction`` /
``calibration_fraction`` / ``test_fraction``); no estimation-affecting numeric
literals appear here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from tdz.config.schema import TDZConfig, ValidationConfig
from tdz.models import QARTruthRecord

__all__ = [
    "FlightGroupKeys",
    "GroupedSplit",
    "GeneralizationSplit",
    "ValidationSplits",
    "group_keys_from_records",
    "make_primary_split",
    "make_generalization_split",
    "make_validation_splits",
]


# ---------------------------------------------------------------------------
# Lightweight per-flight metadata view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlightGroupKeys:
    """The grouping keys needed to build leakage-controlled splits for one flight.

    A lightweight, decoupled view over the fields of
    :class:`~tdz.models.QARTruthRecord` that the splitter needs. Using this view
    (rather than the full truth record) keeps the splitter independent of the
    rest of the truth payload and makes it trivial to unit-test.
    """

    flight_id: str
    tail_number: str
    airport_id: str
    runway_id: str


def group_keys_from_records(
    records: Iterable[QARTruthRecord],
) -> list[FlightGroupKeys]:
    """Project QAR truth records onto the grouping-key view used by the splitter."""
    return [
        FlightGroupKeys(
            flight_id=r.flight_id,
            tail_number=r.tail_number,
            airport_id=r.airport_id,
            runway_id=r.runway_id,
        )
        for r in records
    ]


# ---------------------------------------------------------------------------
# Split result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupedSplit:
    """A primary three-way grouped partition, as lists of ``flight_id``.

    ``group_key`` records which grouping rule produced the split (e.g. ``"tail"``).
    Whole groups are kept intact: no group key appears in more than one of
    ``train`` / ``calibration`` / ``test``.
    """

    group_key: str
    train: tuple[str, ...]
    calibration: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class GeneralizationSplit:
    """A held-out generalization-stress evaluation, as lists of ``flight_id``.

    ``group_key`` is ``"airport"`` or ``"runway"``. Every group value present in
    ``test`` is absent from ``train`` (Req 12.3). This split is a *train/test*
    pair only -- calibration lives with the primary split.
    """

    group_key: str
    train: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class ValidationSplits:
    """The complete set of splits reported by the validation harness.

    ``primary`` is the headline tail-grouped three-way split; the generalization
    evaluations are reported **separately alongside** it, never intersected
    (Req 12.3). A generalization split is ``None`` when its key is not listed in
    ``validation.generalization_evals``.
    """

    primary: GroupedSplit
    held_out_airport: GeneralizationSplit | None
    held_out_runway: GeneralizationSplit | None


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def _key_attr(key: str) -> str:
    """Map a split-key name to the :class:`FlightGroupKeys` attribute holding it."""
    if key == "tail":
        return "tail_number"
    if key == "airport":
        return "airport_id"
    if key == "runway":
        return "runway_id"
    raise ValueError(f"unknown split key '{key}'")


def _group_flights(
    flights: Sequence[FlightGroupKeys], key: str
) -> dict[str, list[str]]:
    """Return ``{group_value -> [flight_id, ...]}`` for the given split key.

    Flight ids within a group preserve input order; groups are consumed in sorted
    key order by callers for determinism.
    """
    attr = _key_attr(key)
    groups: dict[str, list[str]] = defaultdict(list)
    for f in flights:
        groups[getattr(f, attr)].append(f.flight_id)
    return groups


def _partition_counts(n: int, fractions: Sequence[float]) -> list[int]:
    """Split ``n`` items into len(fractions) integer buckets by (normalized) weight.

    Fractions are normalized by their sum, so callers need not pre-normalize.
    Uses the largest-remainder method so the counts always sum to exactly ``n``
    and the allocation is deterministic. When all fractions are zero, everything
    falls into the first bucket.
    """
    total = float(sum(fractions))
    if n <= 0:
        return [0 for _ in fractions]
    if total <= 0.0:
        return [n] + [0 for _ in fractions[1:]]

    weights = [f / total for f in fractions]
    raw = [w * n for w in weights]
    floors = [int(np.floor(x)) for x in raw]
    remainder = n - sum(floors)
    # Distribute the remaining units to the largest fractional remainders.
    order = sorted(
        range(len(fractions)), key=lambda i: (raw[i] - floors[i]), reverse=True
    )
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def _shuffled_group_values(
    groups: dict[str, list[str]], rng: np.random.Generator
) -> list[str]:
    """Deterministically shuffle the group values (sorted first for stability)."""
    values = sorted(groups.keys())
    rng.shuffle(values)
    return values


def _collect(groups: dict[str, list[str]], values: Iterable[str]) -> tuple[str, ...]:
    """Flatten the flight ids of the given group values into a tuple."""
    out: list[str] = []
    for v in values:
        out.extend(groups[v])
    return tuple(out)


# ---------------------------------------------------------------------------
# Public split builders
# ---------------------------------------------------------------------------


def make_primary_split(
    flights: Sequence[FlightGroupKeys],
    validation: ValidationConfig,
    seed: np.random.SeedSequence,
) -> GroupedSplit:
    """Build the primary three-way grouped train/calibration/test split (Req 12.2, 12.4).

    Groups the flights by ``validation.primary_split_key`` (``"tail"`` by default)
    and assigns whole groups to train / calibration / test according to the
    configured fractions. When ``validation.use_calibration_split`` is ``False``
    the calibration fraction is folded into train and the calibration partition
    is empty (a two-way split under the same grouping rule).

    The assignment is deterministic in ``seed``: group values are sorted then
    shuffled with a generator seeded from ``seed``.
    """
    key = validation.primary_split_key
    groups = _group_flights(flights, key)
    rng = np.random.default_rng(seed)
    values = _shuffled_group_values(groups, rng)

    if validation.use_calibration_split:
        fractions = [
            validation.train_fraction,
            validation.calibration_fraction,
            validation.test_fraction,
        ]
    else:
        # No calibration partition: merge its share into train.
        fractions = [
            validation.train_fraction + validation.calibration_fraction,
            0.0,
            validation.test_fraction,
        ]

    n_train, n_cal, _n_test = _partition_counts(len(values), fractions)
    train_vals = values[:n_train]
    cal_vals = values[n_train : n_train + n_cal]
    test_vals = values[n_train + n_cal :]

    return GroupedSplit(
        group_key=key,
        train=_collect(groups, train_vals),
        calibration=_collect(groups, cal_vals),
        test=_collect(groups, test_vals),
    )


def make_generalization_split(
    flights: Sequence[FlightGroupKeys],
    key: str,
    validation: ValidationConfig,
    seed: np.random.SeedSequence,
) -> GeneralizationSplit:
    """Build a held-out-``key`` generalization split (Req 12.3).

    ``key`` is ``"airport"`` or ``"runway"``. A ``test_fraction`` share of the
    distinct group values (by count) is held out entirely into test; every flight
    with a held-out value goes to test and no held-out value appears in train, so
    the test set exercises airports/runways unseen during training. Deterministic
    in ``seed``.
    """
    groups = _group_flights(flights, key)
    rng = np.random.default_rng(seed)
    values = _shuffled_group_values(groups, rng)

    # Hold out `test_fraction` of the distinct group values into test.
    n_train_groups, _n_test_groups = _partition_counts(
        len(values),
        [1.0 - validation.test_fraction, validation.test_fraction],
    )
    train_vals = values[:n_train_groups]
    test_vals = values[n_train_groups:]

    return GeneralizationSplit(
        group_key=key,
        train=_collect(groups, train_vals),
        test=_collect(groups, test_vals),
    )


def make_validation_splits(
    records: Iterable[QARTruthRecord] | Sequence[FlightGroupKeys],
    config: TDZConfig,
) -> ValidationSplits:
    """Build the full set of validation splits for a corpus (Req 12.2, 12.3, 12.4).

    Accepts either raw :class:`~tdz.models.QARTruthRecord` objects or the
    lightweight :class:`FlightGroupKeys` view. Produces the primary tail-grouped
    three-way split plus the held-out-airport and held-out-runway evaluations
    listed in ``config.validation.generalization_evals`` -- each reported
    separately, never intersected into a single split (Req 12.3).

    All partitions are seeded from ``config.pipeline.master_random_seed`` through
    independent spawned seed sequences, so the primary and generalization splits
    are reproducible yet mutually independent (Req 15).
    """
    flights = _as_group_keys(records)
    validation = config.validation

    # Independent, reproducible sub-seeds: primary, airport, runway.
    root = np.random.SeedSequence(config.pipeline.master_random_seed)
    primary_ss, airport_ss, runway_ss = root.spawn(3)

    primary = make_primary_split(flights, validation, primary_ss)

    held_out_airport = None
    held_out_runway = None
    if "airport" in validation.generalization_evals:
        held_out_airport = make_generalization_split(
            flights, "airport", validation, airport_ss
        )
    if "runway" in validation.generalization_evals:
        held_out_runway = make_generalization_split(
            flights, "runway", validation, runway_ss
        )

    return ValidationSplits(
        primary=primary,
        held_out_airport=held_out_airport,
        held_out_runway=held_out_runway,
    )


def _as_group_keys(
    records: Iterable[QARTruthRecord] | Sequence[FlightGroupKeys],
) -> list[FlightGroupKeys]:
    """Normalize the input corpus to a list of :class:`FlightGroupKeys`."""
    materialized = list(records)
    if materialized and isinstance(materialized[0], FlightGroupKeys):
        return materialized  # type: ignore[return-value]
    return group_keys_from_records(materialized)  # type: ignore[arg-type]
