"""YAML configuration loader, schema validation, and defaults resolution.

Pipeline
--------
1. ``load_config(path)`` reads a YAML file into a plain dict.
2. ``build_config(data)`` validates every section against the schema, applies
   defaults for any missing optional parameter (logging a WARNING that names
   the parameter and the applied default), and constructs the typed
   :class:`~tdz.config.schema.TDZConfig` object tree.

Validation is fail-fast: the first violation raises
:class:`~tdz.config.errors.ConfigValidationError` (naming the dotted parameter
path and the violated constraint) and no partially-constructed config is
returned. Required parameters with no sensible default also error.

The fully-resolved configuration (with defaults applied) is retained on the
returned object (``TDZConfig.resolved`` / ``to_dict`` / ``to_yaml``) for
reproducibility and config-hash provenance (Req 20.5).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import yaml

from tdz.config.errors import ConfigError, ConfigValidationError
from tdz.config.models import LeverArm, SourceCapability
from tdz.config.schema import (
    ALLOWED_AIRCRAFT_CLASSES,
    ALLOWED_ELEVATION_DATUMS,
    ALLOWED_ESTIMATORS,
    ALLOWED_GEOID_MODELS,
    ALLOWED_SOURCES,
    ALLOWED_SPLIT_KEYS,
    ClassMedian,
    EstimatorsConfig,
    FusionConfig,
    GeodesyConfig,
    LeverArmsConfig,
    OutputConfig,
    PipelineConfig,
    QualityGatesConfig,
    SignalsConfig,
    TDZConfig,
    TimebaseConfig,
    ValidationConfig,
    VerticalCrossingConfig,
)

logger = logging.getLogger("tdz.config")

# Sentinel marking a required parameter that has no default (missing -> error).
_REQUIRED = object()

# Reserved keys in the lever_arms section that are NOT per-type entries.
_LEVER_ARM_RESERVED = frozenset(
    {"default_strategy", "class_medians", "class_default_widens_ci"}
)

# Canonical class-median defaults (used when the section omits class_medians).
_DEFAULT_CLASS_MEDIANS = {
    "regional": {
        "vertical_offset_m": 3.2,
        "longitudinal_offset_m": 8.0,
        "nominal_touchdown_pitch_deg": 5.0,
    },
    "narrowbody": {
        "vertical_offset_m": 4.2,
        "longitudinal_offset_m": 12.5,
        "nominal_touchdown_pitch_deg": 5.5,
    },
    "widebody": {
        "vertical_offset_m": 5.8,
        "longitudinal_offset_m": 18.0,
        "nominal_touchdown_pitch_deg": 4.5,
    },
}

# Canonical source capability defaults (used when the section omits sources).
_DEFAULT_SOURCES = {
    "aireon": {
        "has_geometric_altitude": True,
        "samples_are_raw": True,
        "async_timestamps": True,
    },
    "fr24": {
        "has_geometric_altitude": False,
        "samples_are_raw": False,
        "async_timestamps": False,
    },
}


# ---------------------------------------------------------------------------
# Low-level scalar validation
# ---------------------------------------------------------------------------


def _type_name(value: Any) -> str:
    return type(value).__name__


def _check_type(path: str, value: Any, type_: type) -> Any:
    """Validate (and lightly coerce) a scalar value against an expected type.

    ``bool`` is never accepted where ``int``/``float`` is expected (even though
    ``bool`` subclasses ``int`` in Python). ``int`` is promoted to ``float``
    where a float is expected.
    """
    if type_ is bool:
        if isinstance(value, bool):
            return value
        raise ConfigValidationError(path, f"expected bool, got {_type_name(value)}")
    if type_ is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigValidationError(path, f"expected int, got {_type_name(value)}")
        return value
    if type_ is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigValidationError(path, f"expected float, got {_type_name(value)}")
        return float(value)
    if type_ is str:
        if not isinstance(value, str):
            raise ConfigValidationError(path, f"expected str, got {_type_name(value)}")
        return value
    raise ConfigValidationError(path, f"unsupported schema type {type_!r}")  # pragma: no cover


def _scalar(
    raw: dict,
    resolved: dict,
    section: str,
    name: str,
    type_: type,
    default: Any,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    choices: Optional[frozenset] = None,
) -> Any:
    """Resolve one scalar parameter: validate, apply default, log warning."""
    path = f"{section}.{name}"

    if name not in raw or raw[name] is None:
        if default is _REQUIRED:
            raise ConfigValidationError(path, "required parameter is missing")
        logger.warning(
            "Config parameter '%s' missing; applying default value %r", path, default
        )
        resolved[name] = default
        return default

    value = _check_type(path, raw[name], type_)

    if minimum is not None and value < minimum:
        raise ConfigValidationError(path, f"must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigValidationError(path, f"must be <= {maximum}, got {value}")
    if choices is not None and value not in choices:
        allowed = ", ".join(sorted(choices))
        raise ConfigValidationError(path, f"must be one of [{allowed}], got '{value}'")

    resolved[name] = value
    return value


def _section_dict(data: dict, name: str, *, required: bool = False) -> dict:
    """Return a section mapping, validating that it is a dict if present."""
    if name not in data or data[name] is None:
        if required:
            raise ConfigValidationError(name, "required section is missing")
        return {}
    value = data[name]
    if not isinstance(value, dict):
        raise ConfigValidationError(name, f"expected mapping, got {_type_name(value)}")
    return value


# ---------------------------------------------------------------------------
# Per-section resolvers
# ---------------------------------------------------------------------------


def _resolve_pipeline(data: dict, resolved: dict) -> PipelineConfig:
    raw = _section_dict(data, "pipeline")
    out: dict = {}
    cfg = PipelineConfig(
        master_random_seed=_scalar(raw, out, "pipeline", "master_random_seed", int, 42, minimum=0, maximum=2**32 - 1),
        ads_b_source=_scalar(raw, out, "pipeline", "ads_b_source", str, "aireon", choices=ALLOWED_SOURCES),
    )
    resolved["pipeline"] = out
    return cfg


def _resolve_timebase(data: dict, resolved: dict) -> TimebaseConfig:
    raw = _section_dict(data, "timebase")
    out: dict = {}
    cfg = TimebaseConfig(
        strategy=_scalar(raw, out, "timebase", "strategy", str, "common_grid", choices=frozenset({"common_grid", "continuous_time"})),
        grid_interval_s=_scalar(raw, out, "timebase", "grid_interval_s", float, 1.0, minimum=0.01, maximum=60.0),
        interpolation_method=_scalar(raw, out, "timebase", "interpolation_method", str, "kinematic", choices=frozenset({"kinematic", "linear"})),
    )
    resolved["timebase"] = out
    return cfg


def _resolve_signals(data: dict, resolved: dict) -> SignalsConfig:
    raw = _section_dict(data, "signals")
    out: dict = {}
    cfg = SignalsConfig(
        smoothing_method=_scalar(raw, out, "signals", "smoothing_method", str, "savgol", choices=frozenset({"savgol", "gp"})),
        savgol_window_samples=_scalar(raw, out, "signals", "savgol_window_samples", int, 7, minimum=3, maximum=101),
        savgol_poly_order=_scalar(raw, out, "signals", "savgol_poly_order", int, 3, minimum=1, maximum=10),
        gp_length_scale_s=_scalar(raw, out, "signals", "gp_length_scale_s", float, 8.0, minimum=0.01, maximum=100.0),
        gp_noise_variance=_scalar(raw, out, "signals", "gp_noise_variance", float, 0.5, minimum=0.0, maximum=100.0),
    )
    resolved["signals"] = out
    return cfg


def _resolve_estimators(data: dict, resolved: dict) -> EstimatorsConfig:
    raw = _section_dict(data, "estimators")
    out: dict = {}
    path = "estimators.enabled"
    default_enabled = [
        "decel_knee",
        "flare_crossing",
        "imm_rts",
        "jerk_onset",
        "pelt",
        "lightgbm",
        "sequence_model",
    ]
    if "enabled" not in raw or raw["enabled"] is None:
        logger.warning(
            "Config parameter '%s' missing; applying default value %r", path, default_enabled
        )
        enabled = list(default_enabled)
    else:
        value = raw["enabled"]
        if not isinstance(value, list):
            raise ConfigValidationError(path, f"expected list, got {_type_name(value)}")
        for item in value:
            if not isinstance(item, str):
                raise ConfigValidationError(path, f"expected str entries, got {_type_name(item)}")
            if item not in ALLOWED_ESTIMATORS:
                raise ConfigValidationError(path, f"unknown estimator '{item}' in estimators.enabled")
        enabled = list(value)
    out["enabled"] = enabled

    cfg = EstimatorsConfig(
        enabled=enabled,
        physics_fallback_threshold=_scalar(raw, out, "estimators", "physics_fallback_threshold", int, 50, minimum=0, maximum=100000),
    )
    resolved["estimators"] = out
    return cfg


def _resolve_fusion(data: dict, resolved: dict) -> FusionConfig:
    raw = _section_dict(data, "fusion")
    out: dict = {}
    cfg = FusionConfig(
        method=_scalar(raw, out, "fusion", "method", str, "stacking", choices=frozenset({"stacking", "weighted_blend"})),
        confidence_threshold_sigma=_scalar(raw, out, "fusion", "confidence_threshold_sigma", float, 5.0, minimum=0.01, maximum=100.0),
        low_confidence_ci_width_ft=_scalar(raw, out, "fusion", "low_confidence_ci_width_ft", float, 600.0, minimum=0.0, maximum=100000.0),
    )
    resolved["fusion"] = out
    return cfg


def _resolve_quality_gates(data: dict, resolved: dict) -> QualityGatesConfig:
    raw = _section_dict(data, "quality_gates")
    out: dict = {}
    cfg = QualityGatesConfig(
        min_samples_near_td=_scalar(raw, out, "quality_gates", "min_samples_near_td", int, 3, minimum=1, maximum=1000),
        max_gap_spanning_td_s=_scalar(raw, out, "quality_gates", "max_gap_spanning_td_s", float, 15.0, minimum=0.0, maximum=600.0),
        min_samples_in_window=_scalar(raw, out, "quality_gates", "min_samples_in_window", int, 3, minimum=1, maximum=1000),
        window_half_width_s=_scalar(raw, out, "quality_gates", "window_half_width_s", float, 30.0, minimum=0.0, maximum=600.0),
        max_excluded_fraction=_scalar(raw, out, "quality_gates", "max_excluded_fraction", float, 0.5, minimum=0.0, maximum=1.0),
        max_longitudinal_accel_g=_scalar(raw, out, "quality_gates", "max_longitudinal_accel_g", float, 1.0, minimum=0.0, maximum=2.0),
        max_lateral_accel_g=_scalar(raw, out, "quality_gates", "max_lateral_accel_g", float, 0.5, minimum=0.0, maximum=2.0),
        max_turn_rate_deg_s=_scalar(raw, out, "quality_gates", "max_turn_rate_deg_s", float, 6.0, minimum=0.0, maximum=90.0),
        duplicate_timestamp_tolerance_s=_scalar(raw, out, "quality_gates", "duplicate_timestamp_tolerance_s", float, 0.1, minimum=0.0, maximum=10.0),
    )
    resolved["quality_gates"] = out
    return cfg


def _resolve_lever_arms(data: dict, resolved: dict) -> LeverArmsConfig:
    raw = _section_dict(data, "lever_arms", required=True)
    out: dict = {}

    # Per-ICAO-type entries are every key that is not a reserved keyword.
    type_keys = [k for k in raw.keys() if k not in _LEVER_ARM_RESERVED]
    if not type_keys:
        raise ConfigValidationError("lever_arms", "lever-arm table has no aircraft-type entries")

    arms: dict[str, LeverArm] = {}
    for icao_type in type_keys:
        entry = raw[icao_type]
        epath = f"lever_arms.{icao_type}"
        if not isinstance(entry, dict):
            raise ConfigValidationError(epath, f"expected mapping, got {_type_name(entry)}")
        eout: dict = {}
        arms[icao_type] = LeverArm(
            icao_type=icao_type,
            vertical_offset_m=_scalar(entry, eout, epath, "vertical_offset_m", float, _REQUIRED, minimum=0.0, maximum=20.0),
            longitudinal_offset_m=_scalar(entry, eout, epath, "longitudinal_offset_m", float, _REQUIRED, minimum=-50.0, maximum=50.0),
            nominal_touchdown_pitch_deg=_scalar(entry, eout, epath, "nominal_touchdown_pitch_deg", float, _REQUIRED, minimum=0.0, maximum=15.0),
            aircraft_class=_scalar(entry, eout, epath, "aircraft_class", str, _REQUIRED, choices=ALLOWED_AIRCRAFT_CLASSES),
            is_class_default=False,
        )
        out[icao_type] = eout

    default_strategy = _scalar(raw, out, "lever_arms", "default_strategy", str, "class_median", choices=frozenset({"class_median"}))

    # class_medians: optional; default to canonical medians.
    cm_path = "lever_arms.class_medians"
    if "class_medians" not in raw or raw["class_medians"] is None:
        logger.warning(
            "Config parameter '%s' missing; applying default value %r", cm_path, _DEFAULT_CLASS_MEDIANS
        )
        cm_raw = _DEFAULT_CLASS_MEDIANS
    else:
        cm_raw = raw["class_medians"]
        if not isinstance(cm_raw, dict):
            raise ConfigValidationError(cm_path, f"expected mapping, got {_type_name(cm_raw)}")

    class_medians: dict[str, ClassMedian] = {}
    cm_out: dict = {}
    for cls_name, cls_entry in cm_raw.items():
        if cls_name not in ALLOWED_AIRCRAFT_CLASSES:
            allowed = ", ".join(sorted(ALLOWED_AIRCRAFT_CLASSES))
            raise ConfigValidationError(f"{cm_path}.{cls_name}", f"unknown aircraft class; must be one of [{allowed}]")
        if not isinstance(cls_entry, dict):
            raise ConfigValidationError(f"{cm_path}.{cls_name}", f"expected mapping, got {_type_name(cls_entry)}")
        mpath = f"{cm_path}.{cls_name}"
        mout: dict = {}
        class_medians[cls_name] = ClassMedian(
            vertical_offset_m=_scalar(cls_entry, mout, mpath, "vertical_offset_m", float, _REQUIRED, minimum=0.0, maximum=20.0),
            longitudinal_offset_m=_scalar(cls_entry, mout, mpath, "longitudinal_offset_m", float, _REQUIRED, minimum=-50.0, maximum=50.0),
            nominal_touchdown_pitch_deg=_scalar(cls_entry, mout, mpath, "nominal_touchdown_pitch_deg", float, _REQUIRED, minimum=0.0, maximum=15.0),
        )
        cm_out[cls_name] = mout
    out["class_medians"] = cm_out

    class_default_widens_ci = _scalar(raw, out, "lever_arms", "class_default_widens_ci", bool, True)

    cfg = LeverArmsConfig(
        arms=arms,
        default_strategy=default_strategy,
        class_medians=class_medians,
        class_default_widens_ci=class_default_widens_ci,
    )
    resolved["lever_arms"] = out
    return cfg


def _resolve_geodesy(data: dict, resolved: dict) -> GeodesyConfig:
    raw = _section_dict(data, "geodesy")
    out: dict = {}
    cfg = GeodesyConfig(
        geoid_model=_scalar(raw, out, "geodesy", "geoid_model", str, "EGM2008", choices=ALLOWED_GEOID_MODELS),
        assume_runway_elevation_datum=_scalar(raw, out, "geodesy", "assume_runway_elevation_datum", str, "MSL", choices=ALLOWED_ELEVATION_DATUMS),
    )
    resolved["geodesy"] = out
    return cfg


def _resolve_vertical_crossing(data: dict, resolved: dict) -> VerticalCrossingConfig:
    raw = _section_dict(data, "vertical_crossing")
    out: dict = {}
    cfg = VerticalCrossingConfig(
        fit_region_upper_ft=_scalar(raw, out, "vertical_crossing", "fit_region_upper_ft", float, 250.0, minimum=0.0, maximum=2000.0),
        fit_region_lower_ft=_scalar(raw, out, "vertical_crossing", "fit_region_lower_ft", float, 0.0, minimum=0.0, maximum=2000.0),
        min_samples_in_fit_region=_scalar(raw, out, "vertical_crossing", "min_samples_in_fit_region", int, 3, minimum=1, maximum=1000),
        residual_bias_trigger_ft=_scalar(raw, out, "vertical_crossing", "residual_bias_trigger_ft", float, 15.0, minimum=0.0, maximum=2000.0),
    )
    resolved["vertical_crossing"] = out
    return cfg


def _resolve_sources(data: dict, resolved: dict) -> dict[str, SourceCapability]:
    path = "sources"
    if "sources" not in data or data["sources"] is None:
        logger.warning(
            "Config section '%s' missing; applying default value %r", path, _DEFAULT_SOURCES
        )
        raw = _DEFAULT_SOURCES
    else:
        raw = data["sources"]
        if not isinstance(raw, dict):
            raise ConfigValidationError(path, f"expected mapping, got {_type_name(raw)}")
        if not raw:
            raise ConfigValidationError(path, "source table has no entries")

    sources: dict[str, SourceCapability] = {}
    out: dict = {}
    for source_name, entry in raw.items():
        if source_name not in ALLOWED_SOURCES:
            allowed = ", ".join(sorted(ALLOWED_SOURCES))
            raise ConfigValidationError(f"{path}.{source_name}", f"unknown source; must be one of [{allowed}]")
        if not isinstance(entry, dict):
            raise ConfigValidationError(f"{path}.{source_name}", f"expected mapping, got {_type_name(entry)}")
        spath = f"{path}.{source_name}"
        sout: dict = {}
        sources[source_name] = SourceCapability(
            source=source_name,
            has_geometric_altitude=_scalar(entry, sout, spath, "has_geometric_altitude", bool, _REQUIRED),
            samples_are_raw=_scalar(entry, sout, spath, "samples_are_raw", bool, _REQUIRED),
            async_timestamps=_scalar(entry, sout, spath, "async_timestamps", bool, _REQUIRED),
        )
        out[source_name] = sout
    resolved["sources"] = out
    return sources


def _resolve_validation(data: dict, resolved: dict) -> ValidationConfig:
    raw = _section_dict(data, "validation")
    out: dict = {}

    primary_split_key = _scalar(raw, out, "validation", "primary_split_key", str, "tail", choices=ALLOWED_SPLIT_KEYS)

    ge_path = "validation.generalization_evals"
    default_ge = ["airport", "runway"]
    if "generalization_evals" not in raw or raw["generalization_evals"] is None:
        logger.warning(
            "Config parameter '%s' missing; applying default value %r", ge_path, default_ge
        )
        generalization_evals = list(default_ge)
    else:
        value = raw["generalization_evals"]
        if not isinstance(value, list):
            raise ConfigValidationError(ge_path, f"expected list, got {_type_name(value)}")
        for item in value:
            if not isinstance(item, str):
                raise ConfigValidationError(ge_path, f"expected str entries, got {_type_name(item)}")
            if item not in ALLOWED_SPLIT_KEYS:
                allowed = ", ".join(sorted(ALLOWED_SPLIT_KEYS))
                raise ConfigValidationError(ge_path, f"unknown split key '{item}'; must be one of [{allowed}]")
        generalization_evals = list(value)
    out["generalization_evals"] = generalization_evals

    cfg = ValidationConfig(
        primary_split_key=primary_split_key,
        generalization_evals=generalization_evals,
        use_calibration_split=_scalar(raw, out, "validation", "use_calibration_split", bool, True),
        min_stratum_size=_scalar(raw, out, "validation", "min_stratum_size", int, 30, minimum=1, maximum=100000),
        cross_source=_scalar(raw, out, "validation", "cross_source", bool, True),
        clock_offset_max_s=_scalar(raw, out, "validation", "clock_offset_max_s", float, 2.0, minimum=0.0, maximum=60.0),
        clock_drift_max_s=_scalar(raw, out, "validation", "clock_drift_max_s", float, 1.0, minimum=0.0, maximum=60.0),
        wrong_runway_lateral_margin_ft=_scalar(raw, out, "validation", "wrong_runway_lateral_margin_ft", float, 50.0, minimum=0.0, maximum=1000.0),
    )
    resolved["validation"] = out
    return cfg


def _resolve_output(data: dict, resolved: dict) -> OutputConfig:
    raw = _section_dict(data, "output")
    out: dict = {}
    cfg = OutputConfig(
        distance_units=_scalar(raw, out, "output", "distance_units", str, "feet", choices=frozenset({"feet", "meters"})),
        speed_units=_scalar(raw, out, "output", "speed_units", str, "knots", choices=frozenset({"knots", "mps"})),
        time_precision_decimals=_scalar(raw, out, "output", "time_precision_decimals", int, 3, minimum=0, maximum=12),
    )
    resolved["output"] = out
    return cfg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_config(data: dict) -> TDZConfig:
    """Validate a raw config mapping and construct a :class:`TDZConfig`.

    Raises :class:`ConfigValidationError` on the first violation; on failure no
    config object is returned (the exception propagates before construction).
    """
    if not isinstance(data, dict):
        raise ConfigValidationError("<root>", f"expected mapping, got {_type_name(data)}")

    resolved: dict = {}
    config = TDZConfig(
        pipeline=_resolve_pipeline(data, resolved),
        timebase=_resolve_timebase(data, resolved),
        signals=_resolve_signals(data, resolved),
        estimators=_resolve_estimators(data, resolved),
        fusion=_resolve_fusion(data, resolved),
        quality_gates=_resolve_quality_gates(data, resolved),
        lever_arms=_resolve_lever_arms(data, resolved),
        geodesy=_resolve_geodesy(data, resolved),
        vertical_crossing=_resolve_vertical_crossing(data, resolved),
        sources=_resolve_sources(data, resolved),
        validation=_resolve_validation(data, resolved),
        output=_resolve_output(data, resolved),
        resolved=resolved,
    )
    return config


def load_config(path: str) -> TDZConfig:
    """Load, validate, and resolve a YAML config file into a :class:`TDZConfig`."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"could not parse YAML config '{path}': {exc}") from exc

    if data is None:
        raise ConfigValidationError("<root>", "configuration file is empty")
    return build_config(data)
