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
"""

from tdz.geo.errors import GeoError, InvalidRunwayReferenceError
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
    "InvalidRunwayReferenceError",
    "GeoError",
]
