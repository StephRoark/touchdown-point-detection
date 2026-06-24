"""Module 6: Time -> Position mapping.

Trajectory interpolation at t_td, pitch-resolved lever-arm correction
(longitudinal*cos(theta) + vertical*sin(theta)), runway centerline projection,
and the wrong-runway lateral-offset gate.

Public API (Task 4 -- geodesy + runway centerline projection):

* :class:`RunwayProjector` / :func:`project_to_runway` -- project a geodetic
  point onto a runway centerline, returning signed along-runway distance and
  lateral offset in meters (Req 11.3, 11.4; Property 1).
* :class:`ProjectedPosition` -- the projection result (meters).
* :func:`validate_runway_reference` -- bounds-check a :class:`RunwayReference`
  (Req 11.5; Property 18).
* :class:`InvalidRunwayReferenceError` -- raised on invalid runway geometry,
  carrying :attr:`FailureReason.INVALID_RUNWAY_REF`.
* :func:`resolve_threshold_elevation_hae` / :class:`DatumResolver` -- unify the
  runway threshold elevation to the HAE datum (geoid-correct an MSL elevation),
  with :func:`lookup_geoid_undulation` as an optional pyproj-backed fallback
  (Req 11.2, 17.2; Property 19).
* :class:`DatumUnresolvedError` -- raised when the vertical datum cannot be
  resolved, carrying :attr:`FailureReason.DATUM_UNRESOLVED`.
"""

from tdz.geo.datum import (
    DatumResolver,
    lookup_geoid_undulation,
    resolve_threshold_elevation_hae,
)
from tdz.geo.errors import (
    DatumUnresolvedError,
    GeoError,
    InvalidRunwayReferenceError,
)
from tdz.geo.projection import (
    ProjectedPosition,
    RunwayProjector,
    project_to_runway,
)
from tdz.geo.runway import validate_runway_reference

__all__ = [
    "ProjectedPosition",
    "RunwayProjector",
    "project_to_runway",
    "validate_runway_reference",
    "resolve_threshold_elevation_hae",
    "DatumResolver",
    "lookup_geoid_undulation",
    "InvalidRunwayReferenceError",
    "DatumUnresolvedError",
    "GeoError",
]
