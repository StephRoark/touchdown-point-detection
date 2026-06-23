"""Tests for runway centerline projection (Task 4.1 / 4.2).

Covers Property 1 (Runway Projection Round-Trip) plus a high-latitude
known-answer test proving geodesic math is actually used, and the projection
edge cases (heading 0/360 wrap, point at threshold, on-centerline point).
"""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pyproj import Geod

from tdz.geo import ProjectedPosition, RunwayProjector, project_to_runway
from tdz.models import RunwayReference

_GEOD = Geod(ellps="WGS84")

# Round-trip tolerance per Requirement 11.4 / Property 1.
_TOL_M = 0.1


def _runway(lat: float, lon: float, heading_deg: float) -> RunwayReference:
    """Build a valid runway reference for projection tests."""
    return RunwayReference(
        threshold_lat=lat,
        threshold_lon=lon,
        heading_deg=heading_deg,
        elevation_m=10.0,
        elevation_datum="HAE",
        geoid_undulation_m=0.0,
        length_m=3000.0,
        width_m=45.0,
        displaced=False,
    )


def _point_at(
    lat: float, lon: float, heading_deg: float, along_m: float, lateral_m: float
) -> tuple[float, float]:
    """Synthesize the geodetic point at (along, lateral) from the threshold.

    Uses :meth:`pyproj.Geod.fwd` directly so the test is independent of the
    projection internals. A point ``along`` m down the runway and ``lateral`` m
    to the right of the landing direction sits at geodesic distance
    ``hypot(along, lateral)`` and azimuth ``heading + atan2(lateral, along)``.
    """
    distance = math.hypot(along_m, lateral_m)
    azimuth = heading_deg + math.degrees(math.atan2(lateral_m, along_m))
    lon2, lat2, _back = _GEOD.fwd(lon, lat, azimuth, distance)
    return lat2, lon2


# ---------------------------------------------------------------------------
# Property 1: Runway Projection Round-Trip
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(
    lat=st.floats(min_value=-80.0, max_value=80.0),
    lon=st.floats(min_value=-179.0, max_value=179.0),
    heading=st.floats(min_value=0.0, max_value=360.0),
    along=st.floats(min_value=0.0, max_value=6000.0),
    lateral=st.floats(min_value=-50.0, max_value=50.0),
)
def test_projection_round_trip(lat, lon, heading, along, lateral):
    """Feature: touchdown-point-detection, Property 1: Runway Projection Round-Trip

    For a point synthesized at a known along-distance D and lateral offset L
    from the threshold, projecting it back recovers D and L within 0.1 m.
    """
    runway = _runway(lat, lon, heading)
    query_lat, query_lon = _point_at(lat, lon, heading, along, lateral)

    result = project_to_runway(runway, query_lat, query_lon)

    assert result.along_runway_distance_m == pytest.approx(along, abs=_TOL_M)
    assert result.lateral_offset_m == pytest.approx(lateral, abs=_TOL_M)


# ---------------------------------------------------------------------------
# High-latitude known-answer test: geodesic vs naive Euclidean
# ---------------------------------------------------------------------------


def _naive_euclidean_along_lateral(
    threshold_lat, threshold_lon, heading_deg, query_lat, query_lon
):
    """Naive flat-earth projection: degrees scaled by a single constant.

    Treats one degree of latitude and one degree of longitude as the same
    number of meters (no latitude/longitude convergence and no ellipsoid),
    i.e. the kind of naive Euclidean computation Requirement 11.4 forbids.
    """
    m_per_deg = 111_320.0
    east = (query_lon - threshold_lon) * m_per_deg
    north = (query_lat - threshold_lat) * m_per_deg
    heading_rad = math.radians(heading_deg)
    along = east * math.sin(heading_rad) + north * math.cos(heading_rad)
    lateral = east * math.cos(heading_rad) - north * math.sin(heading_rad)
    return along, lateral


@pytest.mark.unit
def test_high_latitude_geodesic_beats_naive_euclidean():
    """At 70N, naive Euclidean diverges materially while geodesic round-trips.

    Demonstrates that genuine geodesic math (not flat-earth degrees) is used:
    the implemented projection recovers the known distance within 0.1 m, but the
    naive degrees-based computation is off by many meters at high latitude.
    """
    lat, lon, heading = 70.0, 25.0, 30.0
    along_true, lateral_true = 3000.0, 20.0
    query_lat, query_lon = _point_at(lat, lon, heading, along_true, lateral_true)

    runway = _runway(lat, lon, heading)
    result = project_to_runway(runway, query_lat, query_lon)

    # Geodesic projection round-trips within tolerance.
    assert result.along_runway_distance_m == pytest.approx(along_true, abs=_TOL_M)
    assert result.lateral_offset_m == pytest.approx(lateral_true, abs=_TOL_M)

    # Naive Euclidean degrees-based projection diverges materially (>> 0.1 m).
    naive_along, naive_lateral = _naive_euclidean_along_lateral(
        lat, lon, heading, query_lat, query_lon
    )
    naive_error = math.hypot(naive_along - along_true, naive_lateral - lateral_true)
    assert naive_error > 1.0, (
        f"naive Euclidean error {naive_error:.3f} m should diverge at high latitude"
    )


# ---------------------------------------------------------------------------
# Edge tests (Task 4.4)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_point_at_threshold_is_origin():
    """A query point exactly at the threshold yields along=0 and lateral=0."""
    lat, lon, heading = 40.639801, -73.778900, 43.0
    runway = _runway(lat, lon, heading)
    result = project_to_runway(runway, lat, lon)
    assert result.along_runway_distance_m == pytest.approx(0.0, abs=_TOL_M)
    assert result.lateral_offset_m == pytest.approx(0.0, abs=_TOL_M)


@pytest.mark.unit
def test_point_down_centerline_has_zero_lateral():
    """A point straight down the centerline at distance L has lateral ~= 0."""
    lat, lon, heading = 51.4775, -0.4614, 90.0
    along_true = 2500.0
    query_lat, query_lon = _point_at(lat, lon, heading, along_true, 0.0)
    runway = _runway(lat, lon, heading)
    result = project_to_runway(runway, query_lat, query_lon)
    assert result.along_runway_distance_m == pytest.approx(along_true, abs=_TOL_M)
    assert result.lateral_offset_m == pytest.approx(0.0, abs=_TOL_M)


@pytest.mark.unit
@pytest.mark.parametrize("heading", [0.0, 360.0])
def test_heading_zero_and_360_consistent(heading):
    """Heading 0 and 360 denote the same landing direction (boundary wrap).

    A point due north of the threshold projects to a positive along-distance
    and ~zero lateral offset for both heading conventions.
    """
    lat, lon = 0.0, 0.0
    along_true = 1500.0
    # Place the point due north (azimuth 0) of the threshold.
    lon2, lat2, _back = _GEOD.fwd(lon, lat, 0.0, along_true)
    runway = _runway(lat, lon, heading)
    result = project_to_runway(runway, lat2, lon2)
    assert result.along_runway_distance_m == pytest.approx(along_true, abs=_TOL_M)
    assert result.lateral_offset_m == pytest.approx(0.0, abs=_TOL_M)


@pytest.mark.unit
def test_lateral_sign_positive_is_right_of_landing_direction():
    """Positive lateral offset corresponds to the right of the landing dir."""
    lat, lon, heading = 0.0, 0.0, 0.0  # landing due north
    # A point to the east (right of due-north) should be positive lateral.
    query_lat, query_lon = _point_at(lat, lon, heading, 1000.0, 30.0)
    runway = _runway(lat, lon, heading)
    result = project_to_runway(runway, query_lat, query_lon)
    assert result.lateral_offset_m > 0.0
    assert result.lateral_offset_m == pytest.approx(30.0, abs=_TOL_M)


@pytest.mark.unit
def test_negative_along_before_threshold():
    """A point before the threshold (opposite landing dir) is negative along."""
    lat, lon, heading = 35.0, -106.0, 80.0
    query_lat, query_lon = _point_at(lat, lon, heading, -800.0, 0.0)
    runway = _runway(lat, lon, heading)
    result = project_to_runway(runway, query_lat, query_lon)
    assert result.along_runway_distance_m == pytest.approx(-800.0, abs=_TOL_M)
    assert result.lateral_offset_m == pytest.approx(0.0, abs=_TOL_M)


@pytest.mark.unit
def test_projected_position_is_frozen():
    """ProjectedPosition is an immutable value object."""
    pos = ProjectedPosition(along_runway_distance_m=1.0, lateral_offset_m=2.0)
    with pytest.raises(Exception):
        pos.along_runway_distance_m = 3.0  # type: ignore[misc]


@pytest.mark.unit
def test_projector_reuse_across_points():
    """A constructed projector can be reused for many query points."""
    lat, lon, heading = 48.353, 11.786, 80.0
    projector = RunwayProjector(_runway(lat, lon, heading))
    for along in (0.0, 500.0, 2999.0):
        q_lat, q_lon = _point_at(lat, lon, heading, along, 0.0)
        result = projector.project(q_lat, q_lon)
        assert result.along_runway_distance_m == pytest.approx(along, abs=_TOL_M)
