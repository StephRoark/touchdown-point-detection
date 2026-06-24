"""Source-capability estimator gating (Task 9.2).

Each ADS-B source carries a :class:`~tdz.config.models.SourceCapability`
descriptor (from the config ``sources`` block) that declares, at minimum,
whether the source provides true Geometric_Altitude (HAE) and whether its
position samples are raw observations versus provider-interpolated/smoothed
(Req 8.6). This module turns that descriptor into a structured
:class:`SourceGating` decision listing which estimators are eligible for a
flight and which are excluded -- with a :class:`~tdz.models.FailureReason` code
-- so the fusion layer (a later task) can honor it (Req 8.5, 8.7, 8.8;
Property 20).

Two capability axes are gated here:

* ``has_geometric_altitude is False`` -> every estimator in
  :data:`GEOMETRIC_ALTITUDE_ESTIMATORS` is excluded with
  :attr:`FailureReason.GEOMETRIC_ALT_UNAVAILABLE`. Barometric altitude is NEVER
  substituted into a geometric-altitude field or crossing (Req 8.8); the
  parsers in :mod:`tdz.io.ingest` keep the geometric-altitude array all-``NaN``
  for such a source, and this gating refuses to enable any geometric estimator.

* ``samples_are_raw is False`` (provider-interpolated/smoothed) -> the result's
  :attr:`SourceGating.samples_independent` flag is ``False``, recording that
  downstream noise/fusion models MUST NOT treat the samples as independent raw
  observations (Req 8.7). This module only surfaces the flag; it does not
  implement the noise model.

Everything is driven by the descriptor (no hard-coded per-source behavior), so
the FR24 barometric-only/interpolated ASSUMPTION can be flipped by config alone
once the source's true characteristics are confirmed (Req 8.7).

Geometric-altitude-dependent estimators
---------------------------------------
:data:`GEOMETRIC_ALTITUDE_ESTIMATORS` names the estimator ids that depend on
true geometric altitude (HAE):

* ``flare_crossing`` -- the vertical flare-crossing estimator fits a
  glideslope+flare model directly in HAE and cannot run at all without
  geometric altitude (design "Disabled for any source lacking true geometric
  altitude"). This is a whole-estimator exclusion.
* ``imm_rts`` -- the IMM filter + RTS smoother. The IMM's *geometric-altitude
  updates* are what depend on HAE (Property 20 names "geometric IMM updates").
  A reduced IMM could in principle still run on velocity/position alone, but
  that reduced mode is a fusion-layer concern not yet built; until then this
  gating takes the conservative, Property-20-satisfying choice and excludes the
  whole ``imm_rts`` id so that no geometric IMM update can contribute to the
  fused result. This choice is documented here and can be revisited when the
  IMM estimator and fusion are implemented.

The set is a module constant (documented above) and is injectable as a
parameter so the geometric-dependence policy stays config-driven rather than
hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Optional

from tdz.config.models import SourceCapability
from tdz.models import FailureReason

__all__ = [
    "GEOMETRIC_ALTITUDE_ESTIMATORS",
    "SourceGating",
    "gate_estimators",
]

#: Estimator ids that depend on true geometric altitude (HAE). See the module
#: docstring for the per-estimator rationale (``flare_crossing`` is a whole
#: estimator exclusion; ``imm_rts`` models "geometric IMM altitude updates
#: disabled" as a conservative whole-id exclusion until the reduced-IMM/fusion
#: path exists).
GEOMETRIC_ALTITUDE_ESTIMATORS: Final[frozenset[str]] = frozenset(
    {"flare_crossing", "imm_rts"}
)


@dataclass(frozen=True)
class SourceGating:
    """Per-flight estimator-gating decision derived from a source capability.

    Immutable value object. The exclusion is non-fatal at the flight level (the
    flight still runs through the remaining estimators); it only removes the
    source-ineligible estimators from the fusion set and records why
    (Req 8.5 / Property 20).

    Attributes
    ----------
    source:
        The source identifier the decision was made for (e.g. ``"aireon"``,
        ``"fr24"``).
    has_geometric_altitude:
        Echo of the capability flag; ``False`` means geometric estimators are
        excluded.
    samples_are_raw:
        Echo of the capability flag; ``False`` means provider-interpolated.
    eligible_estimators:
        Estimator ids that may contribute to fusion for this flight, in the
        order they were supplied.
    excluded_estimators:
        Estimator ids excluded for source reasons, in the order they were
        supplied.
    exclusions:
        Tuple of ``(estimator_id, reason_code)`` pairs for every excluded
        estimator (stable order). Currently every exclusion here carries
        :attr:`FailureReason.GEOMETRIC_ALT_UNAVAILABLE`.
    samples_independent:
        ``False`` when ``samples_are_raw`` is ``False`` -- downstream
        noise/independence assumptions MUST NOT treat the samples as independent
        raw observations (Req 8.7). ``True`` otherwise.
    geometric_altitude_estimators:
        The geometric-altitude-dependent estimator set used for this decision
        (defaults to :data:`GEOMETRIC_ALTITUDE_ESTIMATORS`).
    """

    source: str
    has_geometric_altitude: bool
    samples_are_raw: bool
    eligible_estimators: tuple[str, ...]
    excluded_estimators: tuple[str, ...]
    exclusions: tuple[tuple[str, FailureReason], ...]
    samples_independent: bool
    geometric_altitude_estimators: frozenset[str]

    def is_eligible(self, estimator_id: str) -> bool:
        """Return whether ``estimator_id`` may contribute for this flight."""
        return estimator_id in self.eligible_estimators

    def reason_for(self, estimator_id: str) -> Optional[FailureReason]:
        """Return the exclusion reason for ``estimator_id``, or ``None``."""
        for name, reason in self.exclusions:
            if name == estimator_id:
                return reason
        return None


def gate_estimators(
    capability: SourceCapability,
    enabled_estimators: Iterable[str],
    *,
    geometric_altitude_estimators: frozenset[str] = GEOMETRIC_ALTITUDE_ESTIMATORS,
) -> SourceGating:
    """Decide estimator eligibility for a flight from its source capability.

    Parameters
    ----------
    capability:
        The per-source descriptor (from ``config.sources[...]``). Drives the
        whole decision -- there is no hard-coded per-source behavior, so the
        FR24 barometric-only/interpolated assumption is flipped by config alone
        (Req 8.7).
    enabled_estimators:
        The configured set of estimator ids that are otherwise enabled (e.g.
        ``EstimatorsConfig.enabled``). Eligibility is computed as this set minus
        the source-excluded estimators, preserving input order.
    geometric_altitude_estimators:
        The geometric-altitude-dependent estimator ids. Injectable so the
        dependence policy stays config-driven; defaults to
        :data:`GEOMETRIC_ALTITUDE_ESTIMATORS`.

    Returns
    -------
    SourceGating
        The eligible/excluded estimator lists, the per-estimator reason codes,
        and the ``samples_independent`` flag.

    Notes
    -----
    When ``capability.has_geometric_altitude`` is ``False``, every enabled
    estimator that is in ``geometric_altitude_estimators`` is excluded with
    :attr:`FailureReason.GEOMETRIC_ALT_UNAVAILABLE`. Barometric altitude is
    never substituted into a geometric field (Req 8.8); that invariant is
    enforced at parse time (the geometric-altitude array stays ``NaN``) and this
    gating never re-enables a geometric estimator for such a source.
    """
    enabled = list(enabled_estimators)

    eligible: list[str] = []
    excluded: list[str] = []
    exclusions: list[tuple[str, FailureReason]] = []

    for name in enabled:
        if not capability.has_geometric_altitude and name in geometric_altitude_estimators:
            excluded.append(name)
            exclusions.append((name, FailureReason.GEOMETRIC_ALT_UNAVAILABLE))
        else:
            eligible.append(name)

    return SourceGating(
        source=capability.source,
        has_geometric_altitude=capability.has_geometric_altitude,
        samples_are_raw=capability.samples_are_raw,
        eligible_estimators=tuple(eligible),
        excluded_estimators=tuple(excluded),
        exclusions=tuple(exclusions),
        samples_independent=bool(capability.samples_are_raw),
        geometric_altitude_estimators=frozenset(geometric_altitude_estimators),
    )
