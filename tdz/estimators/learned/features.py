"""Per-landing window-feature extraction for the LightGBM estimator (Task 15).

The LightGBM window-feature estimator does **not** consume the variable-length
per-sample channels directly (that is the Task-16 sequence model). Instead each
landing is reduced to a **fixed-length vector of engineered, per-landing summary
features** drawn from the Task-11 signal channels and the physics-derived
segmented groundspeed fit (design "Learned Estimators Detail -> LightGBM Window
Features"; Req 5.3). Feeding physics-derived signals into the learned model lets
it *refine* the physics rather than rediscover it (design key decision "Physics
for trust, learning for accuracy").

The offset reference (the target's well-posed zero)
---------------------------------------------------
The estimator's learning target is a touchdown-time **offset**, not an absolute
epoch time. The offset is measured relative to a **reference time** that is
itself a physics estimate of touchdown: the **segmented-fit groundspeed
breakpoint** (the deceleration knee; :func:`~tdz.signals.segmented.fit_segmented_groundspeed`).
Predicting ``t_td - breakpoint`` makes the target a small, centred residual in
seconds (the correction the data implies on top of the physics knee), which is
far better posed than regressing an absolute epoch second. The reference is
computed identically at train and predict time, so train/predict are consistent.
:func:`extract_window_features` returns this reference on the
:class:`WindowFeatures` it produces.

Because the breakpoint is the offset reference, its *absolute* value is not a
feature; the features instead describe the **shape** of the landing around it
(segment decelerations, speed/decel/jerk summaries, descent and distance
profiles, the on-ground flag's lag past the knee, cadence, sample counts) plus
static context (aircraft type and source as categoricals).

No truth leakage
----------------
Every feature is computable from inputs available at inference (the
:class:`~tdz.models.FlightRecord` and runway geometry). The QAR truth touchdown
time is used **only** to form the training target offset (in the estimator's
``train``), never as a feature here.

Missing values
--------------
LightGBM handles ``NaN`` natively, so features that cannot be computed for a
given source (e.g. height-above-runway / descent rate on a velocity-only FR24
record, or the on-ground-flag lag when no transition time exists) are left as
``NaN`` rather than imputed. The two categorical features (aircraft type, source)
are always present.

Units convention
----------------
SI throughout: seconds, m/s, m/s^2, m/s^3, metres. Times that appear as features
are always **relative** (to the breakpoint or the first sample), never absolute
epoch seconds, so the model never keys on wall-clock time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from tdz.config.schema import SignalsConfig
from tdz.models import FailureReason, FlightRecord
from tdz.signals.features import build_feature_channels
from tdz.signals.segmented import fit_segmented_groundspeed

__all__ = [
    "FEATURE_NAMES",
    "CATEGORICAL_FEATURE_NAMES",
    "CATEGORICAL_FEATURE_INDICES",
    "N_FEATURES",
    "DEFAULT_SIGNALS_CONFIG",
    "SOURCE_CODES",
    "AIRCRAFT_TYPE_VOCAB_SIZE",
    "WindowFeatures",
    "encode_source",
    "encode_aircraft_type",
    "extract_window_features",
]


#: Default smoothing configuration for the derivative channels the features
#: summarise. A 5-sample Savitzky-Golay quadratic is the smallest window the
#: smoother allows (Req 16.2): it suppresses single-sample noise while keeping
#: the deceleration knee sharp. Externalizable -- the estimator may pass the
#: project's resolved :class:`~tdz.config.schema.SignalsConfig` instead.
DEFAULT_SIGNALS_CONFIG: Final[SignalsConfig] = SignalsConfig(
    smoothing_method="savgol",
    savgol_window_samples=5,
    savgol_poly_order=2,
    gp_length_scale_s=8.0,
    gp_noise_variance=0.5,
)

#: Fixed source -> integer-code mapping for the categorical source feature. Both
#: known FR24 spellings collapse to one code; anything else is ``2``.
SOURCE_CODES: Final[dict[str, int]] = {
    "aireon": 0,
    "flightradar24": 1,
    "fr24": 1,
}

#: Size of the aircraft-type hashing vocabulary. The ICAO type designator is
#: hashed deterministically into ``[0, AIRCRAFT_TYPE_VOCAB_SIZE)`` so the
#: categorical code is stable across train/predict without maintaining an
#: explicit vocabulary file.
AIRCRAFT_TYPE_VOCAB_SIZE: Final[int] = 512


def encode_source(ads_b_source: str) -> int:
    """Map an ADS-B source identifier to its stable integer code (see SOURCE_CODES)."""
    return SOURCE_CODES.get(str(ads_b_source).strip().lower(), 2)


def encode_aircraft_type(aircraft_type: str) -> int:
    """Hash an ICAO type designator to a stable code in ``[0, vocab)``.

    Uses a small deterministic FNV-1a hash (not Python's salted ``hash``) so the
    mapping is identical across processes and runs -- a reproducibility
    requirement for the categorical feature (Req 15.1/15.2).
    """
    text = str(aircraft_type).strip().upper().encode("utf-8")
    h = 0x811C9DC5
    for byte in text:
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF
    return int(h % AIRCRAFT_TYPE_VOCAB_SIZE)


#: Ordered names of the engineered features. The order is the column order of the
#: feature matrix and must never be reordered (the trained boosters bind to it).
#: Units are documented per entry below.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    # --- Segmented-fit (physics knee) shape features ---
    "seg_breakpoint_rel_first_velocity_s",  # breakpoint - first velocity time (s)
    "seg_approach_decel_mps2",              # pre-knee segment slope (m/s^2, signed)
    "seg_rollout_decel_mps2",               # post-knee segment slope (m/s^2, signed)
    "seg_slope_drop_mps2",                  # slope_before - slope_after (m/s^2)
    "seg_decel_ratio",                      # |rollout|/|approach| decel (dimensionless)
    "seg_residual_rms_mps",                 # piecewise-fit residual RMS (m/s)
    "seg_approach_speed_mps",               # reconstructed groundspeed at knee (m/s)
    # --- Groundspeed summary (velocity timebase) ---
    "gs_at_breakpoint_mps",                 # observed groundspeed nearest knee (m/s)
    "gs_max_mps",                           # max groundspeed in window (m/s)
    "gs_min_mps",                           # min groundspeed in window (m/s)
    "gs_range_mps",                         # max - min groundspeed (m/s)
    # --- Smoothed deceleration / jerk summary (velocity timebase) ---
    "decel_min_mps2",                       # min (strongest-braking) decel (m/s^2)
    "decel_max_mps2",                       # max decel (m/s^2)
    "decel_argmin_rel_breakpoint_s",        # time of min decel - breakpoint (s)
    "jerk_min_mps3",                        # min smoothed jerk (m/s^3)
    "jerk_max_mps3",                        # max smoothed jerk (m/s^3)
    "jerk_absmax_rel_breakpoint_s",         # time of max |jerk| - breakpoint (s)
    # --- Vertical / distance profile (position timebase; NaN if unavailable) ---
    "descent_rate_mean_mps",                # mean descent rate pre-knee (m/s, +down)
    "height_at_breakpoint_m",               # height above runway nearest knee (m)
    "distance_to_threshold_at_breakpoint_m",  # along-runway distance nearest knee (m)
    "distance_min_abs_m",                   # min |along-runway distance| (m)
    # --- On-ground flag & cadence ---
    "on_ground_transition_rel_breakpoint_s",  # transition - breakpoint (s); NaN if none
    "cadence_velocity_s",                   # median velocity sample spacing (s)
    "cadence_position_s",                   # median position sample spacing (s)
    # --- Sample counts ---
    "n_velocity_samples",                   # count of valid groundspeed samples
    "n_position_samples",                   # count of valid position samples
    "n_geometric_samples",                  # count of finite geometric altitudes
    "has_geometric",                        # 1.0 if any geometric altitude, else 0.0
    # --- Static context (categorical) ---
    "aircraft_type_code",                   # hashed ICAO type designator (categorical)
    "source_code",                          # ADS-B source code (categorical)
)

#: The categorical feature names (handled as LightGBM categoricals).
CATEGORICAL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "aircraft_type_code",
    "source_code",
)

#: Column indices of the categorical features within :data:`FEATURE_NAMES`.
CATEGORICAL_FEATURE_INDICES: Final[tuple[int, ...]] = tuple(
    FEATURE_NAMES.index(name) for name in CATEGORICAL_FEATURE_NAMES
)

#: Number of engineered features (length of every feature vector).
N_FEATURES: Final[int] = len(FEATURE_NAMES)


@dataclass(frozen=True)
class WindowFeatures:
    """A fixed-length engineered feature vector for one landing.

    Attributes
    ----------
    names:
        The feature names, equal to :data:`FEATURE_NAMES` (carried so callers can
        interpret importances without importing the module constant).
    values:
        The feature values (``float`` array of length :data:`N_FEATURES`), in the
        same order as ``names``. May contain ``NaN`` for unavailable features.
    reference_time:
        The offset reference (epoch seconds) -- the segmented-fit breakpoint. The
        training target is ``t_td_truth - reference_time``; at predict time
        ``t_td = reference_time + predicted_offset``.
    reference_kind:
        Identifier of the reference used (``"segmented_breakpoint"`` or, when a
        breakpoint was supplied by the caller, ``"supplied_breakpoint"``).
    categorical_indices:
        Column indices of the categorical features (for the LightGBM Dataset).
    """

    names: tuple[str, ...]
    values: np.ndarray
    reference_time: float
    reference_kind: str
    categorical_indices: tuple[int, ...]


def _nearest_value(times: np.ndarray, values: np.ndarray, target: float) -> float:
    """Return ``values`` at the finite sample whose time is nearest ``target``."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(t) & np.isfinite(v)
    if not np.any(finite):
        return float("nan")
    tf = t[finite]
    vf = v[finite]
    idx = int(np.argmin(np.abs(tf - target)))
    return float(vf[idx])


def _median_spacing(times: np.ndarray) -> float:
    """Median inter-sample spacing of a (possibly unsorted) time array (seconds)."""
    t = np.asarray(times, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return float("nan")
    return float(np.median(np.diff(np.sort(t))))


def _arg_rel(times: np.ndarray, values: np.ndarray, breakpoint: float, *, use_abs: bool) -> float:
    """Time (relative to ``breakpoint``) of the min value, or max |value| if use_abs."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(t) & np.isfinite(v)
    if not np.any(finite):
        return float("nan")
    tf = t[finite]
    vf = v[finite]
    idx = int(np.argmax(np.abs(vf))) if use_abs else int(np.argmin(vf))
    return float(tf[idx] - breakpoint)


def _descent_rate_mean(times: np.ndarray, heights: np.ndarray, breakpoint: float) -> float:
    """Mean descent rate (m/s, positive downward) over the pre-knee samples.

    Computed from the height-above-runway channel: ``-d(height)/dt`` averaged over
    samples up to the breakpoint. Returns ``NaN`` when fewer than two finite
    height samples exist before the knee (e.g. velocity-only sources).
    """
    t = np.asarray(times, dtype=float)
    h = np.asarray(heights, dtype=float)
    finite = np.isfinite(t) & np.isfinite(h)
    if not np.any(finite):
        return float("nan")
    tf = t[finite]
    hf = h[finite]
    order = np.argsort(tf)
    tf = tf[order]
    hf = hf[order]
    pre = tf <= breakpoint
    if int(np.count_nonzero(pre)) >= 2:
        tf = tf[pre]
        hf = hf[pre]
    if tf.size < 2:
        return float("nan")
    dt = tf[-1] - tf[0]
    if dt <= 0.0:
        return float("nan")
    return float(-(hf[-1] - hf[0]) / dt)


def extract_window_features(
    flight: FlightRecord,
    config: Optional[SignalsConfig] = None,
    *,
    breakpoint_time: Optional[float] = None,
    n_segments: int = 2,
) -> tuple[Optional[WindowFeatures], Optional[FailureReason]]:
    """Reduce a landing to a fixed-length engineered feature vector.

    Returns ``(features, None)`` on success or ``(None, reason)`` when the
    landing cannot support the estimator:

    * :attr:`FailureReason.NO_GROUNDSPEED` -- groundspeed is missing entirely.
    * :attr:`FailureReason.INSUFFICIENT_SAMPLES` -- too few valid groundspeed
      samples to fit the segmented (physics-knee) reference.

    Parameters
    ----------
    flight:
        The aligned per-flight record (async timebases preserved).
    config:
        A :class:`~tdz.config.schema.SignalsConfig` for the derivative channels;
        defaults to :data:`DEFAULT_SIGNALS_CONFIG`.
    breakpoint_time:
        Optional offset reference to use instead of the segmented-fit breakpoint
        (e.g. a coarse-bracket centre). When ``None`` the segmented breakpoint is
        fitted and used. Whatever is used is recorded on the result and MUST be
        computed identically at train and predict time.
    n_segments:
        Segments for the segmented groundspeed fit (2 or 3).
    """
    if config is None:
        config = DEFAULT_SIGNALS_CONFIG

    velocity_times = np.asarray(flight.velocity_times, dtype=float)
    gs_kt = np.asarray(flight.groundspeeds, dtype=float)

    if gs_kt.size == 0 or np.all(np.isnan(gs_kt)):
        return None, FailureReason.NO_GROUNDSPEED

    # --- Offset reference: the segmented-fit groundspeed breakpoint (physics knee).
    reference_kind = "segmented_breakpoint"
    try:
        fit = fit_segmented_groundspeed(velocity_times, gs_kt, n_segments=n_segments)
    except ValueError:
        return None, FailureReason.INSUFFICIENT_SAMPLES

    if breakpoint_time is not None:
        reference = float(breakpoint_time)
        reference_kind = "supplied_breakpoint"
    else:
        reference = float(fit.breakpoint_time)

    # Per-segment slopes around the primary breakpoint.
    bp = fit.breakpoint_time
    bp_index = (
        int(np.argmin(np.abs(np.asarray(fit.breakpoint_times) - bp)))
        if fit.breakpoint_times
        else 0
    )
    slope_before = float(fit.slopes_mps2[bp_index])
    slope_after = float(fit.slopes_mps2[bp_index + 1])
    slope_drop = slope_before - slope_after
    approach_decel = slope_before
    rollout_decel = slope_after
    decel_ratio = (
        abs(rollout_decel) / abs(approach_decel) if abs(approach_decel) > 1e-9 else float("nan")
    )

    # Reconstructed approach groundspeed at the knee (from the pre-knee segment).
    finite_v = np.isfinite(velocity_times) & np.isfinite(gs_kt)
    t0 = float(np.min(velocity_times[finite_v])) if np.any(finite_v) else 0.0
    approach_speed_mps = float(fit.intercepts_mps[bp_index] + slope_before * (bp - t0))

    # --- Per-sample channels (reuse Task-11 builder; same breakpoint for smoothing).
    channels = build_feature_channels(flight, config, breakpoint_time=bp)

    gs_mps = channels.groundspeed_mps
    gs_finite = gs_mps[np.isfinite(gs_mps)]
    gs_max = float(np.max(gs_finite)) if gs_finite.size else float("nan")
    gs_min = float(np.min(gs_finite)) if gs_finite.size else float("nan")
    gs_range = gs_max - gs_min if gs_finite.size else float("nan")
    gs_at_bp = _nearest_value(channels.velocity_times, gs_mps, reference)

    decel = channels.deceleration_mps2
    decel_finite = decel[np.isfinite(decel)]
    decel_min = float(np.min(decel_finite)) if decel_finite.size else float("nan")
    decel_max = float(np.max(decel_finite)) if decel_finite.size else float("nan")
    decel_argmin_rel = _arg_rel(channels.velocity_times, decel, reference, use_abs=False)

    jerk = channels.jerk_mps3
    jerk_finite = jerk[np.isfinite(jerk)]
    jerk_min = float(np.min(jerk_finite)) if jerk_finite.size else float("nan")
    jerk_max = float(np.max(jerk_finite)) if jerk_finite.size else float("nan")
    jerk_absmax_rel = _arg_rel(channels.velocity_times, jerk, reference, use_abs=True)

    height = channels.height_above_runway_m
    descent_rate = _descent_rate_mean(channels.position_times, height, reference)
    height_at_bp = _nearest_value(channels.position_times, height, reference)

    distance = channels.distance_to_threshold_m
    distance_at_bp = _nearest_value(channels.position_times, distance, reference)
    dist_finite = distance[np.isfinite(distance)]
    distance_min_abs = float(np.min(np.abs(dist_finite))) if dist_finite.size else float("nan")

    transition = flight.on_ground_transition_time
    transition_rel = (
        float(transition) - reference
        if transition is not None and np.isfinite(transition)
        else float("nan")
    )

    cadence_v = _median_spacing(velocity_times)
    cadence_p = _median_spacing(flight.position_times)

    n_velocity = int(np.count_nonzero(finite_v))
    pos_lat = np.asarray(flight.latitudes, dtype=float)
    n_position = int(np.count_nonzero(np.isfinite(pos_lat)))
    geo = np.asarray(flight.geometric_altitudes, dtype=float)
    n_geometric = int(np.count_nonzero(np.isfinite(geo)))
    has_geometric = 1.0 if n_geometric > 0 else 0.0

    aircraft_code = float(encode_aircraft_type(flight.aircraft_type))
    source_code = float(encode_source(flight.ads_b_source))

    values = np.array(
        [
            float(bp - t0),
            approach_decel,
            rollout_decel,
            slope_drop,
            decel_ratio,
            float(fit.residual_rms_mps),
            approach_speed_mps,
            gs_at_bp,
            gs_max,
            gs_min,
            gs_range,
            decel_min,
            decel_max,
            decel_argmin_rel,
            jerk_min,
            jerk_max,
            jerk_absmax_rel,
            descent_rate,
            height_at_bp,
            distance_at_bp,
            distance_min_abs,
            transition_rel,
            cadence_v,
            cadence_p,
            float(n_velocity),
            float(n_position),
            float(n_geometric),
            has_geometric,
            aircraft_code,
            source_code,
        ],
        dtype=float,
    )
    assert values.size == N_FEATURES, "feature vector length must match FEATURE_NAMES"

    return (
        WindowFeatures(
            names=FEATURE_NAMES,
            values=values,
            reference_time=reference,
            reference_kind=reference_kind,
            categorical_indices=CATEGORICAL_FEATURE_INDICES,
        ),
        None,
    )
