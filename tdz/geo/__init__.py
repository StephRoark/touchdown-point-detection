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

Public API (Task 6 -- pitch-resolved lever-arm correction):

* :class:`LeverArmCorrection` -- the along-runway (``X·cos θ + V·sin θ``) and
  altitude-target (``V``) corrections plus low-confidence / CI-widening
  diagnostics (Req 2.3, 7.2-7.6; Properties 2, 23).
* :func:`compute_lever_arm_correction` / :func:`horizontal_ground_correction`
  -- pitch-resolved geometry for a single :class:`LeverArm`.
* :func:`resolve_lever_arm` -- type-specific -> class-median -> global-median
  resolution (never a worst-case default).
* :func:`resolve_lever_arm_correction` -- resolve + correct + widen in one call.
* :func:`compute_lever_arm_range_widening` -- the class lever-arm-range CI
  inflation magnitude (meters).
* :class:`LeverArmResolutionError` -- raised when no lever arm can be resolved,
  carrying :attr:`FailureReason.MISSING_LEVER_ARM`.

Public API (Task 7 -- position gates):

* :func:`evaluate_position_gates` -- evaluate the wrong-runway lateral-offset
  gate (Req 2.5 / Property 22) and the out-of-bounds along-runway gate (Req 2.4)
  for a :class:`ProjectedPosition`, returning a :class:`PositionGateResult`.
* :class:`PositionGateResult` -- the frozen result carrying the two non-fatal
  flags, the inspected values, the thresholds used, and the triggered
  :class:`~tdz.models.FailureReason` codes
  (:attr:`SUSPECTED_WRONG_RUNWAY`, :attr:`OUT_OF_BOUNDS_POSITION`).
* :func:`is_suspected_wrong_runway` / :func:`is_out_of_bounds` /
  :func:`wrong_runway_lateral_threshold_m` / :func:`resolve_wrong_runway_margin_m`
  -- smaller helpers (the margin is converted from feet with :data:`FT_TO_M`).
* :data:`FT_TO_M` -- the documented feet->meters conversion constant.

Unlike the fatal :class:`InvalidRunwayReferenceError`, the position gates set
non-fatal FLAGS: the estimate is still produced and the value still reported.
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
    LeverArmResolutionError,
)
from tdz.geo.gates import (
    FT_TO_M,
    PositionGateResult,
    evaluate_position_gates,
    is_out_of_bounds,
    is_suspected_wrong_runway,
    resolve_wrong_runway_margin_m,
    wrong_runway_lateral_threshold_m,
)
from tdz.geo.lever_arm import (
    LeverArmCorrection,
    compute_lever_arm_correction,
    compute_lever_arm_range_widening,
    horizontal_ground_correction,
    resolve_lever_arm,
    resolve_lever_arm_correction,
)
from tdz.geo.mapping import (
    TouchdownMapping,
    groundspeed_slope_mps2,
    map_touchdown,
    velocity_samples_within,
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
    "LeverArmCorrection",
    "compute_lever_arm_correction",
    "horizontal_ground_correction",
    "resolve_lever_arm",
    "resolve_lever_arm_correction",
    "compute_lever_arm_range_widening",
    "TouchdownMapping",
    "map_touchdown",
    "groundspeed_slope_mps2",
    "velocity_samples_within",
    "evaluate_position_gates",
    "PositionGateResult",
    "is_suspected_wrong_runway",
    "is_out_of_bounds",
    "wrong_runway_lateral_threshold_m",
    "resolve_wrong_runway_margin_m",
    "FT_TO_M",
    "InvalidRunwayReferenceError",
    "DatumUnresolvedError",
    "LeverArmResolutionError",
    "GeoError",
]
