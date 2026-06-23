"""Threshold-relative runway centerline projection (Task 4.1).

Given a :class:`~tdz.models.RunwayReference` (landing-threshold latitude/longitude
and runway heading in degrees true) and a query point (latitude, longitude), this
module projects the query point onto the runway centerline and returns:

* ``along_runway_distance_m`` -- signed distance along the centerline measured
  from the **landing threshold**. Positive values are *past* the threshold in
  the landing direction; negative values are *before* the threshold
  (Requirement 11.3 sign convention).
* ``lateral_offset_m`` -- perpendicular distance from the centerline. Positive
  values lie to the **right** of the landing direction (starboard); negative
  values lie to the left.

Conventions
-----------
* **Heading** is degrees true, in ``[0, 360]``, with ``0`` = north and angle
  increasing **clockwise** (``90`` = east). ``0`` and ``360`` denote the same
  landing direction.
* **Landing-direction unit vector** in local east/north (ENU) coordinates is
  ``(east, north) = (sin H, cos H)`` for heading ``H`` (radians). The
  to-the-right perpendicular is ``(cos H, -sin H)``.
* All inputs and outputs are SI: latitudes/longitudes in decimal degrees,
  distances in meters. No conversion to feet happens here (that is the output
  boundary, Task 20).

Geodesy
-------
Projection uses geodesic math on the WGS-84 ellipsoid via :class:`pyproj.Geod`,
**not** naive Euclidean distance on raw latitude/longitude (Requirement 11.4).
The query point's geodesic *distance* ``s`` and forward *azimuth* ``alpha`` from
the threshold are computed with :meth:`pyproj.Geod.inv`. These map exactly onto
a local azimuthal-equidistant (ENU) tangent plane centered on the threshold:

    east  = s * sin(alpha)
    north = s * cos(alpha)

Projecting onto the runway frame then reduces to::

    along   = s * cos(alpha - H)
    lateral = s * sin(alpha - H)

Because distance and azimuth from the threshold are preserved exactly along the
geodesic, projection error stays well under 0.1 m over a runway-length baseline
(Requirement 11.4); this is verified by the round-trip property test (P1) and
the high-latitude known-answer test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pyproj import Geod

from tdz.geo.runway import validate_runway_reference
from tdz.models import RunwayReference

__all__ = ["ProjectedPosition", "RunwayProjector", "project_to_runway"]

# WGS-84 ellipsoid; shared and thread-safe for geodesic inverse computations.
_GEOD: Geod = Geod(ellps="WGS84")


@dataclass(frozen=True)
class ProjectedPosition:
    """Result of projecting a point onto a runway centerline (meters).

    Attributes
    ----------
    along_runway_distance_m:
        Signed distance from the landing threshold along the centerline.
        Positive = past the threshold in the landing direction (Req 11.3).
    lateral_offset_m:
        Signed perpendicular distance from the centerline. Positive = to the
        right of the landing direction.
    """

    along_runway_distance_m: float
    lateral_offset_m: float


class RunwayProjector:
    """Projects geodetic points onto a single runway's centerline frame.

    The projector is constructed from a :class:`RunwayReference` whose landing
    (displaced where defined) threshold is the coordinate origin. The reference
    is validated on construction (Requirement 11.5); construct once per runway
    and reuse across many query points.
    """

    def __init__(self, runway: RunwayReference, *, geod: Geod | None = None) -> None:
        validate_runway_reference(runway)
        self._runway = runway
        self._geod = geod if geod is not None else _GEOD
        self._threshold_lat = float(runway.threshold_lat)
        self._threshold_lon = float(runway.threshold_lon)
        # Landing-direction heading in radians (clockwise from north).
        self._heading_rad = math.radians(float(runway.heading_deg))

    @property
    def runway(self) -> RunwayReference:
        """The validated runway reference backing this projector."""
        return self._runway

    def project(self, lat: float, lon: float) -> ProjectedPosition:
        """Project a geodetic point onto the runway centerline frame.

        Parameters
        ----------
        lat, lon:
            Query point latitude/longitude in decimal degrees.

        Returns
        -------
        ProjectedPosition
            Signed along-runway distance and lateral offset in meters.
        """
        # Geodesic inverse from threshold -> query point on the WGS-84 ellipsoid.
        # Returns forward azimuth at the threshold (deg, clockwise from north)
        # and geodesic distance (meters). This is true geodesic math, not a
        # flat-earth approximation (Req 11.4).
        fwd_azimuth_deg, _back_azimuth_deg, distance_m = self._geod.inv(
            self._threshold_lon,
            self._threshold_lat,
            lon,
            lat,
        )

        # Angle of the query point relative to the landing direction. Decompose
        # the polar (distance, azimuth) position in the local azimuthal-
        # equidistant tangent plane onto the runway along/lateral axes.
        delta_rad = math.radians(fwd_azimuth_deg) - self._heading_rad
        along_runway_distance_m = distance_m * math.cos(delta_rad)
        lateral_offset_m = distance_m * math.sin(delta_rad)
        return ProjectedPosition(
            along_runway_distance_m=along_runway_distance_m,
            lateral_offset_m=lateral_offset_m,
        )


def project_to_runway(
    runway: RunwayReference, lat: float, lon: float
) -> ProjectedPosition:
    """Convenience wrapper: project a single point onto ``runway``.

    Equivalent to ``RunwayProjector(runway).project(lat, lon)``. For repeated
    projections against the same runway, construct a :class:`RunwayProjector`
    once and reuse it.
    """
    return RunwayProjector(runway).project(lat, lon)
