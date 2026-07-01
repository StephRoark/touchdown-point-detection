"""Rare-type physics fallback wiring (Task 16; Req 6.2/6.3/6.4; Property 15).

A learned estimator is only trustworthy for aircraft types it has seen enough
of. Requirement 6 makes the physics estimator the **primary** output for *rare*
types -- those with fewer than ``physics_fallback_threshold`` (default 50)
QAR-labeled training flights, counted per type regardless of ADS-B source
(Req 6.3) -- and **always** keeps the interpretable physics anchor (its ``t_td``,
uncertainty, and diagnostics) in the record even when the learned model is
primary (Req 6.2).

What this module provides
-------------------------
* :func:`training_flight_counts` -- per-aircraft-type counts from the QAR truth
  set (the count that decides rare vs common).
* :class:`PrimaryWithAnchor` -- the selection result: the chosen primary
  estimate plus the always-present physics anchor and (for common types) the
  learned estimate.
* :class:`RareTypePhysicsFallback` -- the selector that applies the policy:

  - **Rare type (count < threshold):** physics is primary; the learned estimate
    is **omitted** entirely (not merely down-weighted). If the physics estimator
    cannot produce a valid estimate for the rare-type flight, the result is a
    **low-confidence, no-touchdown** record (``t_td = NaN``) rather than a
    fallback to the learned model (Req 6.4) -- we never let an
    under-trained learned model speak for a rare type.
  - **Common type (count >= threshold):** the learned estimate is primary when
    it succeeds; if it fails, the physics anchor takes over (it is not a rare
    type, so physics-primary is acceptable). Either way the physics anchor is in
    the record.

Property 15
-----------
For any type with fewer than the threshold training flights, the selector marks
the physics estimator as the primary contributor and omits the learned estimate,
with the physics anchor present -- the invariant the property test asserts.

This module is deliberately free of any torch dependency: the rare-type path
never invokes the learned model, so the policy (and its property test) runs in a
torch-less environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping, Optional, Sequence

from tdz.estimators.physics.base import (
    CONFIDENCE_FAILED,
    CONFIDENCE_LOW,
)
from tdz.models import BaseEstimator, FailureReason, FlightRecord, QARTruthRecord, TDEstimate

__all__ = [
    "DEFAULT_PHYSICS_FALLBACK_THRESHOLD",
    "PHYSICS_PRIMARY",
    "LEARNED_PRIMARY",
    "training_flight_counts",
    "PrimaryWithAnchor",
    "RareTypePhysicsFallback",
]

#: Default rare-type threshold (Req 6.3; config ``estimators.physics_fallback_threshold``).
DEFAULT_PHYSICS_FALLBACK_THRESHOLD: Final[int] = 50

#: Identifiers for which family produced the primary output.
PHYSICS_PRIMARY: Final[str] = "physics"
LEARNED_PRIMARY: Final[str] = "learned"


def training_flight_counts(truths: Sequence[QARTruthRecord]) -> dict[str, int]:
    """Count QAR-labeled training flights per aircraft type (Req 6.3).

    Counted per ICAO type designator regardless of ADS-B source, matching the
    requirement's "counted per aircraft type regardless of ADS-B source".
    """
    counts: dict[str, int] = {}
    for truth in truths:
        counts[truth.aircraft_type] = counts.get(truth.aircraft_type, 0) + 1
    return counts


def _is_usable(estimate: Optional[TDEstimate]) -> bool:
    """True when an estimate exists, did not fail, and carries a finite ``t_td``."""
    return (
        estimate is not None
        and estimate.confidence != CONFIDENCE_FAILED
        and estimate.t_td is not None
        and math.isfinite(estimate.t_td)
    )


@dataclass(frozen=True)
class PrimaryWithAnchor:
    """Selected primary estimate plus the always-present physics anchor.

    Attributes
    ----------
    t_td:
        The primary touchdown time (epoch seconds); ``NaN`` when no touchdown is
        produced (rare type whose physics estimate failed -- Req 6.4).
    sigma_t:
        The primary 1-sigma uncertainty (seconds).
    confidence:
        ``"normal"`` | ``"low-confidence"`` | ``"failed"``.
    reason_code:
        Machine-readable reason for low-confidence / no-touchdown, else ``None``.
    primary_source:
        :data:`PHYSICS_PRIMARY` or :data:`LEARNED_PRIMARY` -- which family is the
        primary contributor (the field Property 15 inspects).
    primary_method:
        The ``method_name`` of the primary estimator.
    physics_anchor:
        The physics estimator's :class:`TDEstimate`, **always** present (Req 6.2).
    learned_estimate:
        The learned estimator's :class:`TDEstimate`, or ``None`` when omitted
        (always omitted for rare types -- Req 6.3/6.4).
    aircraft_type:
        The flight's ICAO type designator.
    n_training_flights:
        Training-flight count for this type (the rare/common decision input).
    is_rare_type:
        ``True`` when ``n_training_flights < threshold``.
    touchdown_omitted:
        ``True`` when no touchdown estimate is produced (Req 6.4 rare-type miss).
    """

    t_td: float
    sigma_t: float
    confidence: str
    reason_code: Optional[str]
    primary_source: str
    primary_method: str
    physics_anchor: TDEstimate
    learned_estimate: Optional[TDEstimate]
    aircraft_type: str
    n_training_flights: int
    is_rare_type: bool
    touchdown_omitted: bool


class RareTypePhysicsFallback:
    """Apply the Requirement-6 rare-type physics-primary policy.

    Parameters
    ----------
    physics_estimator:
        The interpretable physics anchor / rare-type primary (e.g.
        :class:`~tdz.estimators.physics.decel_knee.DecelKneeEstimator`).
    learned_estimator:
        The learned estimator (e.g.
        :class:`~tdz.estimators.learned.sequence_model.SequenceModelEstimator`).
    training_type_counts:
        Per-aircraft-type training-flight counts (see :func:`training_flight_counts`).
        When ``None`` the counts are read from ``learned_estimator.training_type_counts``
        if available, else treated as empty (every type rare).
    threshold:
        Rare-type cutoff; defaults to :data:`DEFAULT_PHYSICS_FALLBACK_THRESHOLD`.
        Pass ``config.estimators.physics_fallback_threshold`` to honour config.
    """

    def __init__(
        self,
        physics_estimator: BaseEstimator,
        learned_estimator: Optional[BaseEstimator] = None,
        *,
        training_type_counts: Optional[Mapping[str, int]] = None,
        threshold: int = DEFAULT_PHYSICS_FALLBACK_THRESHOLD,
    ) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        self.physics_estimator = physics_estimator
        self.learned_estimator = learned_estimator
        if training_type_counts is None:
            training_type_counts = getattr(
                learned_estimator, "training_type_counts", {}
            )
        self.training_type_counts: dict[str, int] = dict(training_type_counts or {})
        self.threshold = int(threshold)

    def training_count(self, aircraft_type: str) -> int:
        """Training-flight count for ``aircraft_type`` (0 if never seen)."""
        return int(self.training_type_counts.get(aircraft_type, 0))

    def is_rare_type(self, aircraft_type: str) -> bool:
        """``True`` when the type has fewer than ``threshold`` training flights."""
        return self.training_count(aircraft_type) < self.threshold

    def select(self, flight: FlightRecord) -> PrimaryWithAnchor:
        """Choose the primary estimate for ``flight`` and attach the physics anchor."""
        physics_anchor = self.physics_estimator.estimate(flight)
        n_flights = self.training_count(flight.aircraft_type)
        rare = self.is_rare_type(flight.aircraft_type)

        if rare:
            return self._select_rare(flight, physics_anchor, n_flights)
        return self._select_common(flight, physics_anchor, n_flights)

    def _select_rare(
        self, flight: FlightRecord, physics_anchor: TDEstimate, n_flights: int
    ) -> PrimaryWithAnchor:
        """Rare type: physics is primary; learned is omitted (Req 6.3/6.4)."""
        physics_method = getattr(self.physics_estimator, "method_name", "physics")

        if _is_usable(physics_anchor):
            return PrimaryWithAnchor(
                t_td=float(physics_anchor.t_td),
                sigma_t=float(physics_anchor.sigma_t),
                confidence=physics_anchor.confidence,
                reason_code=physics_anchor.diagnostics.get("reason_code"),
                primary_source=PHYSICS_PRIMARY,
                primary_method=physics_method,
                physics_anchor=physics_anchor,
                learned_estimate=None,
                aircraft_type=flight.aircraft_type,
                n_training_flights=n_flights,
                is_rare_type=True,
                touchdown_omitted=False,
            )

        # Req 6.4: physics failed for a rare type -> low-confidence, omit the
        # touchdown estimate; do NOT fall back to the under-trained learned model.
        reason = physics_anchor.diagnostics.get("reason_code")
        return PrimaryWithAnchor(
            t_td=float("nan"),
            sigma_t=float("inf"),
            confidence=CONFIDENCE_LOW,
            reason_code=reason or FailureReason.INSUFFICIENT_SAMPLES.value,
            primary_source=PHYSICS_PRIMARY,
            primary_method=physics_method,
            physics_anchor=physics_anchor,
            learned_estimate=None,
            aircraft_type=flight.aircraft_type,
            n_training_flights=n_flights,
            is_rare_type=True,
            touchdown_omitted=True,
        )

    def _select_common(
        self, flight: FlightRecord, physics_anchor: TDEstimate, n_flights: int
    ) -> PrimaryWithAnchor:
        """Common type: learned is primary when usable, else physics anchor."""
        learned = (
            self.learned_estimator.estimate(flight)
            if self.learned_estimator is not None
            else None
        )

        if _is_usable(learned):
            learned_method = getattr(
                self.learned_estimator, "method_name", "learned"
            )
            return PrimaryWithAnchor(
                t_td=float(learned.t_td),
                sigma_t=float(learned.sigma_t),
                confidence=learned.confidence,
                reason_code=learned.diagnostics.get("reason_code"),
                primary_source=LEARNED_PRIMARY,
                primary_method=learned_method,
                physics_anchor=physics_anchor,
                learned_estimate=learned,
                aircraft_type=flight.aircraft_type,
                n_training_flights=n_flights,
                is_rare_type=False,
                touchdown_omitted=False,
            )

        # Learned unavailable/failed on a common type: physics anchor is primary.
        physics_method = getattr(self.physics_estimator, "method_name", "physics")
        if _is_usable(physics_anchor):
            return PrimaryWithAnchor(
                t_td=float(physics_anchor.t_td),
                sigma_t=float(physics_anchor.sigma_t),
                confidence=physics_anchor.confidence,
                reason_code=physics_anchor.diagnostics.get("reason_code"),
                primary_source=PHYSICS_PRIMARY,
                primary_method=physics_method,
                physics_anchor=physics_anchor,
                learned_estimate=learned,
                aircraft_type=flight.aircraft_type,
                n_training_flights=n_flights,
                is_rare_type=False,
                touchdown_omitted=False,
            )

        # Neither produced a usable estimate.
        return PrimaryWithAnchor(
            t_td=float("nan"),
            sigma_t=float("inf"),
            confidence=CONFIDENCE_FAILED,
            reason_code=FailureReason.ALL_ESTIMATORS_FAILED.value,
            primary_source=PHYSICS_PRIMARY,
            primary_method=physics_method,
            physics_anchor=physics_anchor,
            learned_estimate=learned,
            aircraft_type=flight.aircraft_type,
            n_training_flights=n_flights,
            is_rare_type=False,
            touchdown_omitted=True,
        )
