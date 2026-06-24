"""Shared scaffolding for the physics estimators (Task 12.4).

This module provides the pieces every physics estimator shares:

* :class:`PhysicsEstimator` -- a concrete :class:`~tdz.models.BaseEstimator`
  subclass that implements the common :meth:`estimate` contract and applies the
  **on-ground-flag upper bound** to whatever candidate ``t_td`` a subclass
  produces. Subclasses implement :meth:`_raw_estimate` (the physics) and set
  :attr:`method_name`; the bound is enforced uniformly here so no estimator can
  forget it (Req 18.1-18.4; Property 5).
* :func:`apply_on_ground_bound` -- the bound itself, exposed as a free function
  so it can be unit-tested and reused by fusion (Task 18) and the property test.
* :func:`failed_estimate` / :func:`make_estimate` -- small constructors for the
  :class:`~tdz.models.TDEstimate` contract so confidence strings and the
  ``reason_code`` diagnostics key are written consistently across estimators.

On-ground-flag upper bound (Requirement 18)
-------------------------------------------
The ADS-B on-ground flag transitions with a variable *delay* after the true
touchdown, so it is only ever an **upper time-bound** on ``t_td`` -- never the
answer (Req 18.1, design key decision "On-ground flag is an upper bound on
``t_td``"). The bound mechanism therefore:

* **never** outputs the transition time itself as ``t_td`` (Req 18.1);
* constrains ``t_td <= transition`` (Req 18.2);
* **clamps** any candidate at or after the transition back to just before it
  (Req 18.3) -- a clamped candidate lands at ``transition - guard`` where
  :data:`ON_GROUND_BOUND_GUARD_S` is a tiny strict-inequality guard (so the
  output is strictly < the transition, satisfying 18.1 and 18.2 together), and
* gives the flag **zero weight** otherwise: when a candidate is already before
  the transition the flag does not move it at all (Req 18.4). The flag is a
  bracket, not a measurement that is averaged in.

When a candidate is clamped, the reported ``sigma_t`` is widened to at least the
distance the estimate was moved (in quadrature) so the lost information is
reflected honestly, and the clamp is recorded in diagnostics
(``on_ground_clamped``, ``on_ground_bound``, ``pre_clamp_t_td``).

Units: ``t_td`` and ``sigma_t`` are seconds (epoch seconds and a 1-sigma width).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Optional

from tdz.models import BaseEstimator, FailureReason, FlightRecord, TDEstimate

__all__ = [
    "ON_GROUND_BOUND_GUARD_S",
    "CONFIDENCE_NORMAL",
    "CONFIDENCE_LOW",
    "CONFIDENCE_FAILED",
    "OnGroundBoundResult",
    "apply_on_ground_bound",
    "make_estimate",
    "failed_estimate",
    "PhysicsEstimator",
]

#: Strict-inequality guard (seconds). A candidate ``t_td`` at or after the
#: on-ground transition is clamped to ``transition - ON_GROUND_BOUND_GUARD_S``
#: so the reported time is strictly *before* the transition (Req 18.1 forbids
#: outputting the transition time itself; Req 18.2 requires ``t_td <=
#: transition``). This is a numerical guard to keep the inequality strict, not
#: an estimation tunable -- its magnitude is negligible relative to the
#: sub-second resolution the system targets.
ON_GROUND_BOUND_GUARD_S: Final[float] = 1e-3

#: The three confidence classes an estimator may report on a :class:`TDEstimate`
#: (the fusion layer maps "failed" to the output-record "no-estimate" class).
CONFIDENCE_NORMAL: Final[str] = "normal"
CONFIDENCE_LOW: Final[str] = "low-confidence"
CONFIDENCE_FAILED: Final[str] = "failed"


def _is_finite(value: Optional[float]) -> bool:
    """Return ``True`` when ``value`` is a finite real number."""
    return value is not None and not math.isnan(value) and not math.isinf(value)


@dataclass(frozen=True)
class OnGroundBoundResult:
    """Result of applying the on-ground upper bound to a candidate ``t_td``.

    Attributes
    ----------
    t_td:
        The bounded touchdown time (seconds). Equal to the input candidate when
        no clamp was needed; otherwise ``transition - ON_GROUND_BOUND_GUARD_S``.
    clamped:
        ``True`` when the candidate was at or after the transition and had to be
        moved back (Req 18.3).
    bound:
        The on-ground transition time used as the upper bound, or ``None`` when
        no transition time was available (the bound is then a no-op).
    pre_clamp_t_td:
        The original candidate before clamping (for diagnostics).
    """

    t_td: float
    clamped: bool
    bound: Optional[float]
    pre_clamp_t_td: float


def apply_on_ground_bound(
    candidate_t_td: float,
    on_ground_transition_time: Optional[float],
    *,
    guard_s: float = ON_GROUND_BOUND_GUARD_S,
) -> OnGroundBoundResult:
    """Constrain a candidate ``t_td`` to the on-ground-flag upper bound.

    Implements Requirement 18 exactly:

    * If no transition time is available (``None``/non-finite), the candidate is
      returned unchanged -- the flag contributes nothing (Req 18.4: zero weight).
    * If the candidate is strictly before the transition, it is returned
      unchanged (the flag is a bracket, not a measurement -- Req 18.4).
    * If the candidate is at or after the transition, it is clamped to
      ``transition - guard_s`` (Req 18.3) so the result is strictly less than
      the transition (Req 18.1, 18.2).

    Parameters
    ----------
    candidate_t_td:
        The estimator's raw candidate touchdown time (seconds).
    on_ground_transition_time:
        The on-ground-flag transition time (seconds), or ``None``.
    guard_s:
        Strict-inequality guard; defaults to :data:`ON_GROUND_BOUND_GUARD_S`.

    Returns
    -------
    OnGroundBoundResult
        The bounded time plus whether a clamp occurred and the bound used.
    """
    if not _is_finite(candidate_t_td):
        return OnGroundBoundResult(
            t_td=candidate_t_td,
            clamped=False,
            bound=on_ground_transition_time if _is_finite(on_ground_transition_time) else None,
            pre_clamp_t_td=candidate_t_td,
        )

    if not _is_finite(on_ground_transition_time):
        # No usable bound: the flag has zero weight (Req 18.4).
        return OnGroundBoundResult(
            t_td=candidate_t_td, clamped=False, bound=None, pre_clamp_t_td=candidate_t_td
        )

    transition = float(on_ground_transition_time)
    if candidate_t_td < transition:
        # Already inside the bracket: the flag does not move it (Req 18.4).
        return OnGroundBoundResult(
            t_td=candidate_t_td, clamped=False, bound=transition, pre_clamp_t_td=candidate_t_td
        )

    # At or after the transition: clamp to strictly before it (Req 18.1-18.3).
    return OnGroundBoundResult(
        t_td=transition - guard_s,
        clamped=True,
        bound=transition,
        pre_clamp_t_td=candidate_t_td,
    )


def make_estimate(
    *,
    t_td: float,
    sigma_t: float,
    confidence: str,
    method_name: str,
    diagnostics: dict,
    reason: Optional[FailureReason] = None,
) -> TDEstimate:
    """Build a :class:`TDEstimate`, writing the reason code into diagnostics.

    The ``reason_code`` diagnostics key is set to ``reason.value`` (or ``None``)
    so every estimator surfaces a low-confidence/failure reason consistently.
    """
    diag = dict(diagnostics)
    diag["reason_code"] = reason.value if reason is not None else None
    return TDEstimate(
        t_td=float(t_td),
        sigma_t=float(sigma_t),
        confidence=confidence,
        diagnostics=diag,
        method_name=method_name,
    )


def failed_estimate(
    method_name: str, reason: FailureReason, diagnostics: Optional[dict] = None
) -> TDEstimate:
    """Build a ``confidence="failed"`` estimate carrying ``reason``.

    A failed estimator does **not** raise; it returns this sentinel so the
    fusion layer can exclude it with the reason logged (design Error Propagation
    Strategy, module 4a). ``t_td`` is ``NaN`` and ``sigma_t`` is ``inf``.
    """
    return make_estimate(
        t_td=float("nan"),
        sigma_t=float("inf"),
        confidence=CONFIDENCE_FAILED,
        method_name=method_name,
        diagnostics=diagnostics or {},
        reason=reason,
    )


class PhysicsEstimator(BaseEstimator):
    """Concrete base for the physics estimators with the on-ground bound baked in.

    Subclasses implement :meth:`_raw_estimate` (the physics that produces a
    candidate :class:`TDEstimate`) and set the class attribute
    :attr:`method_name`. :meth:`estimate` calls the subclass and then applies
    :func:`apply_on_ground_bound` uniformly, so the Requirement-18 upper bound is
    impossible to skip (Property 5). Failed estimates pass through untouched (a
    ``NaN`` candidate cannot exceed any bound).
    """

    #: Estimator identifier (e.g. ``"decel_knee"``); set by each subclass.
    method_name: str = "physics"

    def name(self) -> str:
        """Return the estimator identifier (matches the configured estimator id)."""
        return self.method_name

    def _raw_estimate(self, flight: FlightRecord) -> TDEstimate:
        """Produce the raw candidate estimate; implemented by subclasses."""
        raise NotImplementedError

    def estimate(self, flight: FlightRecord) -> TDEstimate:
        """Produce a touchdown estimate, bounded by the on-ground flag (Req 18)."""
        raw = self._raw_estimate(flight)
        return self._bound_to_on_ground(raw, flight)

    def _bound_to_on_ground(self, estimate: TDEstimate, flight: FlightRecord) -> TDEstimate:
        """Apply the on-ground upper bound and record the clamp in diagnostics."""
        if estimate.confidence == CONFIDENCE_FAILED:
            # Nothing to bound: a failed estimate has no usable t_td.
            return estimate

        bound = apply_on_ground_bound(estimate.t_td, flight.on_ground_transition_time)

        diagnostics = dict(estimate.diagnostics)
        diagnostics["on_ground_bound"] = bound.bound
        diagnostics["on_ground_clamped"] = bound.clamped

        sigma_t = estimate.sigma_t
        if bound.clamped:
            diagnostics["pre_clamp_t_td"] = bound.pre_clamp_t_td
            # Widen sigma_t by the clamp distance (in quadrature) so the lost
            # information is reflected honestly in the reported uncertainty.
            moved = abs(bound.pre_clamp_t_td - bound.t_td)
            sigma_t = float(math.hypot(sigma_t, moved))

        return TDEstimate(
            t_td=bound.t_td,
            sigma_t=sigma_t,
            confidence=estimate.confidence,
            diagnostics=diagnostics,
            method_name=estimate.method_name,
        )
