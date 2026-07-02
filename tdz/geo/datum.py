"""Vertical datum unification: orthometric (MSL) -> HAE (Task 5).

ADS-B ``Geometric_Altitude`` is height above the WGS-84 ellipsoid (HAE).
Published runway/airport elevations are almost always *orthometric* (height
above mean sea level / the geoid). The two datums differ by the local **geoid
undulation** ``N`` -- commonly -15 to -35 m over the continental United States,
and up to roughly +/-100 m globally. Comparing an MSL runway elevation directly
against HAE geometric altitude would inject that tens-of-metres offset straight
into the vertical-crossing estimator, so the datum MUST be unified to HAE before
any crossing (Requirement 11.2, 17.2; Property 19).

Sign convention
---------------
This module implements exactly::

    h(HAE) = H(orthometric / MSL) + N(geoid undulation)

where ``N`` is the EGM2008 geoid undulation at the threshold: the height of the
geoid above the ellipsoid. Over the continental US the geoid lies *below* the
ellipsoid, so ``N`` is typically **negative** (~ -15 to -35 m); adding a
negative ``N`` to an MSL elevation therefore yields a *smaller* HAE value. A
runway already tagged ``"HAE"`` is returned unchanged -- the undulation is
**never** applied a second time.

Separation of concerns
-----------------------
This is the *deterministic* geoid/datum correction only. It is kept completely
separate from any empirical residual sensor-bias estimation (Requirement 17.3,
Task 12.2): folding a deterministic datum offset into an empirical "bias" term
would hide errors and could absorb real flare dynamics. Nothing in this module
touches ADS-B samples or estimates a bias -- it converts a single published
elevation between vertical datums.

Geoid undulation source
------------------------
The primary path uses the ``geoid_undulation_m`` already supplied on the
:class:`~tdz.models.RunwayReference` (the design states this is the EGM2008
undulation at the threshold). This path requires **no external grids or
network**. An optional helper (:func:`lookup_geoid_undulation`) can look the
undulation up from a configured geoid model via :mod:`pyproj` when one is not
supplied; that path degrades gracefully when EGM2008 grids are absent (see the
function docstring) and is never required for the supplied-undulation path.

Validation
----------
Field bounds for ``elevation_m`` / ``elevation_datum`` are the responsibility of
:func:`tdz.geo.runway.validate_runway_reference` (Requirement 11.5). This module
assumes a (possibly) pre-validated runway and focuses on the datum conversion;
it independently rejects a missing/unrecognized datum, a missing/non-finite
undulation for an MSL elevation, and any non-finite converted output by raising
:class:`DatumUnresolvedError` (``reason_code = FailureReason.DATUM_UNRESOLVED``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Optional

from tdz.geo.errors import DatumUnresolvedError
from tdz.models import RunwayReference

__all__ = [
    "DatumResolver",
    "resolve_threshold_elevation_hae",
    "lookup_geoid_undulation",
    "DATUM_HAE",
    "DATUM_MSL",
]

#: Canonical datum tags. A height-above-ellipsoid elevation needs no correction;
#: an orthometric (mean-sea-level) elevation is geoid-corrected to HAE.
DATUM_HAE: Final[str] = "HAE"
DATUM_MSL: Final[str] = "MSL"


def _coerce_finite_float(value: object) -> Optional[float]:
    """Return ``value`` as a finite float, or ``None`` if it is not usable.

    ``None``, booleans, non-numeric types, ``NaN`` and infinities all map to
    ``None`` so callers can treat "no usable value" uniformly.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _normalize_datum(raw: object) -> Optional[str]:
    """Normalize a datum tag to ``"HAE"``/``"MSL"`` or ``None`` if unrecognized."""
    if not isinstance(raw, str):
        return None
    tag = raw.strip().upper()
    if tag in (DATUM_HAE, DATUM_MSL):
        return tag
    return None


def lookup_geoid_undulation(
    lat: float, lon: float, *, geoid_model: str = "EGM2008"
) -> Optional[float]:
    """Look up the geoid undulation ``N`` (metres) at ``(lat, lon)`` via pyproj.

    This is an **optional** convenience used only when a runway does not carry a
    supplied ``geoid_undulation_m``. It builds a vertical-transform pipeline
    (ellipsoidal height -> orthometric height) and reads back the undulation as
    the height difference at zero ellipsoidal height.

    The required geoid grids (e.g. the EGM2008 ``us_nga_egm08_25.tif`` grid) may
    not be installed or downloadable in every environment. Rather than crashing,
    this function **degrades gracefully**: any failure to import pyproj, build
    the transform, locate the grid, or produce a finite result returns ``None``
    so the caller can fall back to the supplied undulation or raise a clear
    :class:`DatumUnresolvedError`. It never raises.

    Parameters
    ----------
    lat, lon:
        Threshold latitude/longitude in decimal degrees.
    geoid_model:
        Geoid model name (informational; the active pyproj/PROJ data determines
        which grid is actually used).

    Returns
    -------
    float or None
        The undulation ``N`` in metres (geoid height above the ellipsoid), or
        ``None`` when no grid is available / the lookup could not be completed.
    """
    try:
        from pyproj.transformer import TransformerGroup

        # Ellipsoidal height (EPSG:4979, WGS-84 3D) -> orthometric height on the
        # EGM2008 geoid (EPSG:3855). The vertical shift at h_ellipsoidal = 0 is
        # -N, so N = h_ellipsoidal - h_orthometric evaluated at h = 0.
        group = TransformerGroup(4979, 3855)
        if not group.transformers:
            return None
        # A transformer is unusable if its grids are not available locally.
        # NEVER fall back to a transformer with missing grids: pyproj may then
        # apply a degraded (lower-accuracy or null) vertical shift and return a
        # plausible-looking but wrong undulation -- exactly the silent
        # tens-of-metres vertical bias this module exists to prevent. Degrade
        # to None instead so the caller raises DatumUnresolvedError.
        available = [
            t
            for t in group.transformers
            if getattr(t, "is_network_enabled", False)
            or _transformer_grids_available(t)
        ]
        if not available:
            return None
        transformer = available[0]
        _lon_out, _lat_out, h_orthometric = transformer.transform(lon, lat, 0.0)
        if h_orthometric is None:
            return None
        undulation = 0.0 - float(h_orthometric)
        if math.isnan(undulation) or math.isinf(undulation):
            return None
        return undulation
    except Exception:
        # Missing pyproj, missing grids, or any transform failure: degrade.
        return None


def _transformer_grids_available(transformer: object) -> bool:
    """Best-effort probe of whether a transformer's grids are present locally."""
    try:
        operations = getattr(transformer, "operations", None) or []
        for op in operations:
            for grid in getattr(op, "grids", None) or []:
                if not getattr(grid, "available", True):
                    return False
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class DatumResolver:
    """Resolves a runway threshold elevation to the HAE datum.

    The resolver is deterministic and holds only the optional fallback policy
    (which geoid model to consult, and which datum to assume for an untagged
    elevation). Construct once from a :class:`~tdz.config.schema.GeodesyConfig`
    and reuse across runways.

    Parameters
    ----------
    geoid_model:
        Geoid model used for the optional pyproj undulation lookup when a runway
        does not supply ``geoid_undulation_m`` (default ``"EGM2008"``).
    assume_elevation_datum:
        Datum assumed when a runway's ``elevation_datum`` is missing/empty. When
        ``None`` (the default), an untagged datum is treated as unresolved.
    allow_geoid_lookup:
        Whether to attempt the pyproj geoid-grid lookup as a fallback. Defaults
        to ``True``; the lookup degrades to ``None`` when grids are absent.
    """

    geoid_model: str = "EGM2008"
    assume_elevation_datum: Optional[str] = None
    allow_geoid_lookup: bool = True

    @classmethod
    def from_config(cls, geodesy_config: object) -> "DatumResolver":
        """Build a resolver from a :class:`GeodesyConfig`-like object.

        Reads ``geoid_model`` and ``assume_runway_elevation_datum`` when present
        so datum policy is externalized to configuration rather than hard-coded.
        ``geodesy_config`` may be ``None``, in which case defaults are used.
        """
        if geodesy_config is None:
            return cls()
        geoid_model = getattr(geodesy_config, "geoid_model", None) or "EGM2008"
        assume = _normalize_datum(
            getattr(geodesy_config, "assume_runway_elevation_datum", None)
        )
        return cls(geoid_model=str(geoid_model), assume_elevation_datum=assume)

    def resolve(self, runway: RunwayReference) -> float:
        """Return ``runway``'s threshold elevation in HAE metres.

        See :func:`resolve_threshold_elevation_hae` for the full contract.
        """
        elevation = _coerce_finite_float(getattr(runway, "elevation_m", None))
        if elevation is None:
            raise DatumUnresolvedError(
                "elevation_m: missing or non-finite; cannot resolve vertical datum"
            )

        datum = _normalize_datum(getattr(runway, "elevation_datum", None))
        if datum is None:
            datum = self.assume_elevation_datum
        if datum is None:
            raw = getattr(runway, "elevation_datum", None)
            raise DatumUnresolvedError(
                "elevation_datum: unrecognized or missing datum tag "
                f"({raw!r}); expected 'HAE' or 'MSL' and no configured default"
            )

        if datum == DATUM_HAE:
            # Already HAE: return unchanged. The undulation is NOT applied.
            hae = elevation
        else:
            # Orthometric (MSL): h(HAE) = H(MSL) + N(undulation).
            undulation = _coerce_finite_float(
                getattr(runway, "geoid_undulation_m", None)
            )
            if undulation is None and self.allow_geoid_lookup:
                undulation = lookup_geoid_undulation(
                    float(getattr(runway, "threshold_lat", float("nan"))),
                    float(getattr(runway, "threshold_lon", float("nan"))),
                    geoid_model=self.geoid_model,
                )
            if undulation is None:
                raise DatumUnresolvedError(
                    "geoid_undulation_m: MSL elevation requires a finite geoid "
                    "undulation, but none was supplied and no geoid model was "
                    f"available (geoid_model={self.geoid_model!r})"
                )
            hae = elevation + undulation

        if math.isnan(hae) or math.isinf(hae):
            raise DatumUnresolvedError(
                f"resolved HAE elevation is not finite ({hae}); datum unresolved"
            )
        return hae


def resolve_threshold_elevation_hae(
    runway: RunwayReference, geodesy_config: object = None
) -> float:
    """Convert a runway threshold elevation to the HAE datum (metres).

    * If ``elevation_datum == "HAE"``: the elevation is already height above the
      ellipsoid and is returned unchanged (the undulation is NOT added).
    * If ``elevation_datum == "MSL"``: the orthometric elevation is converted to
      HAE by adding the local geoid undulation ``N`` --
      ``hae = elevation_m + geoid_undulation_m`` (Requirement 11.2). ``N`` is the
      EGM2008 undulation at the threshold and is typically negative over the
      continental US.

    An MSL elevation is **never** returned/compared as if it were HAE without the
    correction (Requirement 11.2).

    Parameters
    ----------
    runway:
        The runway whose threshold elevation is converted. Field bounds are
        assumed to have been checked by
        :func:`~tdz.geo.runway.validate_runway_reference`; this function still
        independently rejects a missing/unrecognized datum, a missing/non-finite
        undulation for an MSL elevation, and a non-finite result.
    geodesy_config:
        Optional :class:`~tdz.config.schema.GeodesyConfig`-like object supplying
        ``geoid_model`` and ``assume_runway_elevation_datum`` (used only as
        fallbacks). When ``None``, defaults are used and an untagged datum is
        treated as unresolved.

    Returns
    -------
    float
        The threshold elevation in HAE metres.

    Raises
    ------
    DatumUnresolvedError
        When the datum cannot be resolved (missing/unrecognized datum tag; MSL
        elevation with no finite undulation and no available geoid model; or a
        non-finite converted result). Carries
        :attr:`~tdz.models.FailureReason.DATUM_UNRESOLVED`.
    """
    return DatumResolver.from_config(geodesy_config).resolve(runway)
