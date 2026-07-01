"""Cross-source generalization evaluation (Task 22.3, Req 12.9).

This module measures how much accuracy the system loses when a model trained on
one ADS-B source is applied to the *other* source -- the source-transfer penalty
(Req 12.9). To make that penalty attributable to the **source** rather than to a
different mix of flights, the evaluation is restricted to the physical landings
observed by **both** feeds (the "both-feeds intersection"): the same underlying
landing is held constant, so a measured accuracy difference reflects the source,
not a different flight population.

The four arms
-------------
With two sources ``A`` and ``B`` there are four train/test arms::

    (A, A)  same-source  -- model trained on A, tested on A's flights
    (B, B)  same-source  -- model trained on B, tested on B's flights
    (A, B)  cross-source -- model trained on A, tested on B's flights  (A -> B)
    (B, A)  cross-source -- model trained on B, tested on A's flights  (B -> A)

All four are computed over the *same* shared-landing intersection so the arms
are directly comparable.

Accuracy drop vs same-source
----------------------------
The source-transfer penalty is reported per direction by comparing a cross arm
against the same-source arm **on the same test source** (same physical test
flights, only the training source differs):

* direction ``B -> A``: cross arm ``(B, A)`` vs same-source arm ``(A, A)``
* direction ``A -> B``: cross arm ``(A, B)`` vs same-source arm ``(B, B)``

The drop is reported both as an RMSE delta in feet (``cross - same``; positive
means the cross-trained model is worse) and as a percentage of the same-source
RMSE. Holding the test flights fixed across the two arms is what isolates the
source effect: any RMSE change is due solely to which source the model was
trained on.

Pairing across feeds
--------------------
Two feeds observe the *same physical landing*, so both are validated against the
same :class:`~tdz.models.QARTruthRecord`. The shared landing identity therefore
defaults to that record's ``flight_id`` (see :func:`default_landing_key`); a
custom key can be supplied for feeds keyed differently. Metric math is delegated
entirely to :func:`tdz.validation.metrics.compute_metrics` -- this module only
selects the intersection, runs the four arms, and computes the drop.

Units
-----
All distances are feet, consistent with
:mod:`tdz.validation.metrics` (single SI->feet boundary via ``M_TO_FT``). This
module introduces no estimation-affecting numeric literals; whether the
cross-source evaluation is reported at all is gated by
``config.validation.cross_source`` (see :func:`evaluate_cross_source`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable, Mapping, Sequence

import numpy as np

from tdz.config.schema import ValidationConfig
from tdz.models import ValidationMetrics
from tdz.validation.metrics import FlightEvaluation, compute_metrics

__all__ = [
    "CrossSourceArm",
    "CrossSourceDirection",
    "CrossSourceReport",
    "default_landing_key",
    "shared_landings",
    "evaluate_cross_source",
]

#: The two ADS-B sources compared by the cross-source evaluation.
SOURCE_AIREON: str = "aireon"
SOURCE_FR24: str = "flightradar24"


# ---------------------------------------------------------------------------
# Landing identity (pairing across feeds)
# ---------------------------------------------------------------------------


def default_landing_key(ev: FlightEvaluation) -> Hashable:
    """Shared physical-landing identity used to pair a landing across feeds.

    Both feeds that observe one physical landing are validated against the same
    QAR truth record, so its ``flight_id`` identifies the landing independently
    of which source produced the estimate. Supply a custom key function to
    :func:`evaluate_cross_source` when feeds are keyed differently.
    """
    return ev.truth.flight_id


# ---------------------------------------------------------------------------
# Report containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossSourceArm:
    """Metrics for one train/test arm over the shared-landing intersection.

    ``train_source`` is the source the model was trained on; ``test_source`` is
    the source the test flights came from. ``metrics`` are the standard
    :class:`~tdz.models.ValidationMetrics` computed on the both-feeds subset.
    """

    train_source: str
    test_source: str
    n_flights: int
    metrics: ValidationMetrics


@dataclass(frozen=True)
class CrossSourceDirection:
    """One transfer direction with its same-source reference and the drop.

    ``cross`` is the arm trained on the *other* source; ``same_source`` is the
    arm trained on ``test_source`` itself. Both are evaluated on the identical
    (shared-landing) test flights of ``test_source``, so the deltas isolate the
    source-transfer penalty. ``rmse_drop_ft`` is ``cross - same`` in feet
    (positive = cross-trained model is worse); ``rmse_drop_pct`` expresses it as
    a percentage of the same-source RMSE.
    """

    train_source: str
    test_source: str
    cross: ValidationMetrics
    same_source: ValidationMetrics
    rmse_drop_ft: float
    rmse_drop_pct: float


@dataclass(frozen=True)
class CrossSourceReport:
    """The full cross-source evaluation over the both-feeds intersection.

    ``n_shared_landings`` is the number of physical landings observed by both
    feeds (the intersection the four arms are restricted to). ``same_source``
    holds the two same-source arms and ``cross_source`` the two cross-source
    arms; ``directions`` pairs each cross arm with its same-source reference and
    carries the accuracy drop.
    """

    source_a: str
    source_b: str
    n_shared_landings: int
    same_source: tuple[CrossSourceArm, ...]
    cross_source: tuple[CrossSourceArm, ...]
    directions: tuple[CrossSourceDirection, ...]


# ---------------------------------------------------------------------------
# Intersection selection
# ---------------------------------------------------------------------------


def shared_landings(
    arms: Mapping[tuple[str, str], Sequence[FlightEvaluation]],
    source_a: str,
    source_b: str,
    landing_key: Callable[[FlightEvaluation], Hashable],
) -> set[Hashable]:
    """Return the landing keys observed by **both** feeds (the intersection).

    A landing is "observed by" a source if that source appears as the
    ``test_source`` of some arm containing an evaluation for the landing. The
    returned set is the intersection of the landings observed by ``source_a`` and
    those observed by ``source_b``; landings present in only one feed are
    excluded so the four arms compare like-for-like flights.
    """
    observed_a = _observed_landings(arms, source_a, landing_key)
    observed_b = _observed_landings(arms, source_b, landing_key)
    return observed_a & observed_b


def _observed_landings(
    arms: Mapping[tuple[str, str], Sequence[FlightEvaluation]],
    test_source: str,
    landing_key: Callable[[FlightEvaluation], Hashable],
) -> set[Hashable]:
    """Landing keys seen in any arm whose ``test_source`` matches."""
    seen: set[Hashable] = set()
    for (_train, test), evals in arms.items():
        if test == test_source:
            for ev in evals:
                seen.add(landing_key(ev))
    return seen


def _filter_to(
    evals: Sequence[FlightEvaluation],
    keep: set[Hashable],
    landing_key: Callable[[FlightEvaluation], Hashable],
) -> list[FlightEvaluation]:
    """Keep only evaluations whose landing key is in ``keep``."""
    return [ev for ev in evals if landing_key(ev) in keep]


# ---------------------------------------------------------------------------
# Drop arithmetic
# ---------------------------------------------------------------------------


def _rmse_drop(cross_rmse: float, same_rmse: float) -> tuple[float, float]:
    """Return ``(delta_ft, pct)`` for a cross vs same-source RMSE comparison.

    ``delta_ft`` is ``cross - same`` (positive means the cross-trained model is
    worse). ``pct`` normalizes the delta by the same-source RMSE; it is ``nan``
    when the same-source RMSE is not a positive, finite number (nothing to
    compare against).
    """
    delta = float(cross_rmse) - float(same_rmse)
    if np.isfinite(same_rmse) and same_rmse > 0.0:
        pct = delta / float(same_rmse) * 100.0
    else:
        pct = float("nan")
    return delta, pct


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_cross_source(
    arms: Mapping[tuple[str, str], Sequence[FlightEvaluation]],
    config: ValidationConfig,
    *,
    source_a: str = SOURCE_AIREON,
    source_b: str = SOURCE_FR24,
    landing_key: Callable[[FlightEvaluation], Hashable] = default_landing_key,
) -> CrossSourceReport | None:
    """Run the cross-source generalization evaluation (Req 12.9).

    ``arms`` maps each ``(train_source, test_source)`` pair to that arm's
    per-flight evaluations. All four arms must be supplied: the two same-source
    arms ``(source_a, source_a)`` and ``(source_b, source_b)`` and the two
    cross-source arms ``(source_a, source_b)`` and ``(source_b, source_a)``.
    Training happens elsewhere; this function consumes each arm's already-paired
    :class:`~tdz.validation.metrics.FlightEvaluation` outputs.

    The evaluation restricts every arm to the physical landings observed by both
    feeds (see :func:`shared_landings`), computes metrics per arm via
    :func:`~tdz.validation.metrics.compute_metrics`, and reports the accuracy
    drop of each cross direction against the same-source arm on the *same* test
    source, isolating the source-transfer penalty.

    Returns ``None`` when ``config.cross_source`` is ``False`` (the evaluation is
    gated off and not reported). Raises :class:`ValueError` when any of the four
    required arms is missing.
    """
    if not config.cross_source:
        return None

    required = [
        (source_a, source_a),
        (source_b, source_b),
        (source_a, source_b),
        (source_b, source_a),
    ]
    missing = [arm for arm in required if arm not in arms]
    if missing:
        raise ValueError(
            "evaluate_cross_source requires all four train/test arms; missing: "
            + ", ".join(f"(train={t}, test={e})" for t, e in missing)
        )

    keep = shared_landings(arms, source_a, source_b, landing_key)

    def _arm(train: str, test: str) -> CrossSourceArm:
        members = _filter_to(arms[(train, test)], keep, landing_key)
        metrics = compute_metrics(members, stratum_key=f"train={train}/test={test}")
        return CrossSourceArm(
            train_source=train,
            test_source=test,
            n_flights=metrics.n_flights,
            metrics=metrics,
        )

    same_aa = _arm(source_a, source_a)
    same_bb = _arm(source_b, source_b)
    cross_ab = _arm(source_a, source_b)  # A -> B (tested on B)
    cross_ba = _arm(source_b, source_a)  # B -> A (tested on A)

    # Pair each cross direction with the same-source arm on the SAME test source.
    directions: list[CrossSourceDirection] = []
    for cross_arm, same_arm in ((cross_ba, same_aa), (cross_ab, same_bb)):
        delta, pct = _rmse_drop(
            cross_arm.metrics.distance_rmse_ft, same_arm.metrics.distance_rmse_ft
        )
        directions.append(
            CrossSourceDirection(
                train_source=cross_arm.train_source,
                test_source=cross_arm.test_source,
                cross=cross_arm.metrics,
                same_source=same_arm.metrics,
                rmse_drop_ft=delta,
                rmse_drop_pct=pct,
            )
        )

    return CrossSourceReport(
        source_a=source_a,
        source_b=source_b,
        n_shared_landings=len(keep),
        same_source=(same_aa, same_bb),
        cross_source=(cross_ab, cross_ba),
        directions=tuple(directions),
    )
