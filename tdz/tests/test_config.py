"""Tests for the configuration module (Task 3).

Covers:
- Property 12 (Configuration Schema Validation): random invalid mutations of a
  valid config are rejected with a ``ConfigValidationError`` whose message names
  the offending parameter and the violated constraint.
- Unit tests: a valid config loads and round-trips all sections; missing
  optional parameters fall back to defaults with a logged WARNING; the canonical
  ``config/tdz_config.yaml`` loads and validates; LeverArm/SourceCapability
  objects build correctly; an empty lever-arm table is rejected.
"""

from __future__ import annotations

import copy
import logging
import string
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from tdz.config import (
    ALLOWED_ESTIMATORS,
    ConfigValidationError,
    LeverArm,
    SourceCapability,
    TDZConfig,
    build_config,
    load_config,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

CANONICAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "tdz_config.yaml"
)


def _base_dict() -> dict:
    """A valid configuration mapping loaded from the canonical YAML."""
    with open(CANONICAL_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _set(cfg: dict, section: str, key: str, value) -> None:
    cfg[section][key] = value


# Float-typed scalar parameters with their declared max (for out-of-range tests).
_RANGE_CASES = [
    ("quality_gates", "max_longitudinal_accel_g", 2.0),
    ("quality_gates", "max_lateral_accel_g", 2.0),
    ("quality_gates", "max_excluded_fraction", 1.0),
    ("fusion", "confidence_threshold_sigma", 100.0),
    ("timebase", "grid_interval_s", 60.0),
    ("validation", "clock_offset_max_s", 60.0),
]

# Numeric scalar parameters (for wrong-type tests; a string always fails).
_NUMERIC_PATHS = [
    ("pipeline", "master_random_seed"),
    ("timebase", "grid_interval_s"),
    ("signals", "savgol_window_samples"),
    ("signals", "gp_length_scale_s"),
    ("fusion", "confidence_threshold_sigma"),
    ("quality_gates", "max_longitudinal_accel_g"),
    ("vertical_crossing", "fit_region_upper_ft"),
    ("validation", "min_stratum_size"),
]


# ---------------------------------------------------------------------------
# Property 12: Configuration Schema Validation
# ---------------------------------------------------------------------------


@pytest.mark.property
@given(data=st.data())
def test_p12_invalid_mutations_rejected(data):
    """Feature: touchdown-point-detection, Property 12: Configuration Schema Validation

    For any config parameter value that fails schema validation (wrong type,
    out-of-range value, or unknown estimator name), building the config raises
    ConfigValidationError naming the parameter and the violated constraint, and
    no config object is returned.
    """
    base = _base_dict()
    kind = data.draw(
        st.sampled_from(["wrong_type", "out_of_range", "unknown_estimator"])
    )
    cfg = copy.deepcopy(base)

    if kind == "wrong_type":
        section, key = data.draw(st.sampled_from(_NUMERIC_PATHS))
        bad = data.draw(st.text(min_size=1, max_size=8))
        _set(cfg, section, key, bad)
        expected_path = f"{section}.{key}"
        constraint_tokens = ("expected",)

    elif kind == "out_of_range":
        section, key, max_val = data.draw(st.sampled_from(_RANGE_CASES))
        bad = data.draw(
            st.floats(
                min_value=max_val + 0.001,
                max_value=max_val + 1.0e6,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        _set(cfg, section, key, bad)
        expected_path = f"{section}.{key}"
        constraint_tokens = ("must be <=",)

    else:  # unknown_estimator
        name = data.draw(
            st.text(alphabet=string.ascii_lowercase + "_", min_size=1, max_size=12).filter(
                lambda s: s not in ALLOWED_ESTIMATORS
            )
        )
        cfg["estimators"]["enabled"] = list(cfg["estimators"]["enabled"]) + [name]
        expected_path = "estimators.enabled"
        constraint_tokens = ("unknown estimator",)

    with pytest.raises(ConfigValidationError) as excinfo:
        build_config(cfg)

    message = str(excinfo.value)
    assert expected_path in message, f"message did not name the parameter: {message!r}"
    assert any(tok in message for tok in constraint_tokens), (
        f"message did not state the constraint: {message!r}"
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_canonical_config_loads_and_validates(caplog):
    """The shipped canonical config loads, validates, and applies no defaults."""
    with caplog.at_level(logging.WARNING, logger="tdz.config"):
        cfg = load_config(str(CANONICAL_CONFIG_PATH))

    assert isinstance(cfg, TDZConfig)
    # Canonical config is complete -> no default-applied warnings.
    warnings = [r for r in caplog.records if r.name == "tdz.config"]
    assert warnings == [], f"unexpected default warnings: {[w.message for w in warnings]}"


@pytest.mark.unit
def test_valid_config_round_trips_all_sections():
    """A valid config exposes every section with the expected exact values."""
    cfg = build_config(_base_dict())

    assert cfg.pipeline.master_random_seed == 42
    assert cfg.pipeline.ads_b_source == "aireon"
    assert cfg.timebase.strategy == "common_grid"
    assert cfg.timebase.grid_interval_s == 1.0
    assert cfg.timebase.interpolation_method == "kinematic"
    assert cfg.signals.smoothing_method == "savgol"
    assert cfg.signals.savgol_window_samples == 7
    assert cfg.signals.savgol_poly_order == 3
    assert cfg.signals.gp_length_scale_s == 8.0
    assert cfg.signals.gp_noise_variance == 0.5
    assert "decel_knee" in cfg.estimators.enabled
    assert cfg.estimators.physics_fallback_threshold == 50
    assert cfg.fusion.method == "stacking"
    assert cfg.fusion.confidence_threshold_sigma == 5.0
    assert cfg.fusion.low_confidence_ci_width_ft == 600.0
    assert cfg.fusion.low_confidence_ci_width_s == 8.0
    assert cfg.fusion.disagreement_threshold_s == 3.0
    assert cfg.quality_gates.max_longitudinal_accel_g == 1.0
    assert cfg.quality_gates.duplicate_timestamp_tolerance_s == 0.1
    assert cfg.geodesy.geoid_model == "EGM2008"
    assert cfg.geodesy.assume_runway_elevation_datum == "MSL"
    assert cfg.vertical_crossing.fit_region_upper_ft == 250.0
    assert cfg.validation.primary_split_key == "tail"
    assert cfg.validation.generalization_evals == ["airport", "runway"]
    assert cfg.output.distance_units == "feet"
    assert cfg.output.speed_units == "knots"
    assert cfg.output.time_precision_decimals == 3


@pytest.mark.unit
def test_resolved_config_round_trips_through_yaml():
    """to_yaml() output re-loads to an identical resolved config (reproducibility)."""
    cfg = build_config(_base_dict())
    reserialized = cfg.to_yaml()
    reloaded = build_config(yaml.safe_load(reserialized))
    assert reloaded.to_dict() == cfg.to_dict()


@pytest.mark.unit
def test_lever_arm_and_source_objects_build():
    """lever_arms entries become LeverArm objects; sources become SourceCapability."""
    cfg = build_config(_base_dict())

    b738 = cfg.lever_arms.arms["B738"]
    assert isinstance(b738, LeverArm)
    assert b738.icao_type == "B738"
    assert b738.vertical_offset_m == 4.2
    assert b738.longitudinal_offset_m == 12.5
    assert b738.nominal_touchdown_pitch_deg == 5.5
    assert b738.aircraft_class == "narrowbody"
    assert b738.is_class_default is False

    assert cfg.lever_arms.default_strategy == "class_median"
    assert cfg.lever_arms.class_default_widens_ci is True
    assert cfg.lever_arms.class_medians["widebody"].vertical_offset_m == 5.8

    aireon = cfg.sources["aireon"]
    assert isinstance(aireon, SourceCapability)
    assert aireon.source == "aireon"
    assert aireon.has_geometric_altitude is True
    fr24 = cfg.sources["fr24"]
    assert fr24.has_geometric_altitude is False
    assert fr24.samples_are_raw is False


@pytest.mark.unit
def test_missing_optional_param_applies_default_and_warns(caplog):
    """A missing optional scalar falls back to its default with a logged WARNING."""
    cfg_dict = _base_dict()
    del cfg_dict["signals"]["gp_noise_variance"]

    with caplog.at_level(logging.WARNING, logger="tdz.config"):
        cfg = build_config(cfg_dict)

    assert cfg.signals.gp_noise_variance == 0.5  # applied default
    messages = [r.getMessage() for r in caplog.records if r.name == "tdz.config"]
    assert any(
        "signals.gp_noise_variance" in m and "0.5" in m for m in messages
    ), f"no default-applied warning found: {messages}"


@pytest.mark.unit
def test_missing_optional_section_applies_all_defaults_and_warns(caplog):
    """An omitted optional section resolves entirely from defaults with warnings."""
    cfg_dict = _base_dict()
    del cfg_dict["output"]

    with caplog.at_level(logging.WARNING, logger="tdz.config"):
        cfg = build_config(cfg_dict)

    assert cfg.output.distance_units == "feet"
    assert cfg.output.speed_units == "knots"
    assert cfg.output.time_precision_decimals == 3
    messages = [r.getMessage() for r in caplog.records if r.name == "tdz.config"]
    assert any("output.distance_units" in m for m in messages)


@pytest.mark.unit
def test_empty_lever_arm_table_rejected():
    """A lever_arms section with no aircraft-type entries is rejected."""
    cfg_dict = _base_dict()
    cfg_dict["lever_arms"] = {
        "default_strategy": "class_median",
        "class_default_widens_ci": True,
    }
    with pytest.raises(ConfigValidationError) as excinfo:
        build_config(cfg_dict)
    assert "lever_arms" in str(excinfo.value)


@pytest.mark.unit
def test_missing_lever_arms_section_rejected():
    """The lever_arms section is required (cannot apply correction without it)."""
    cfg_dict = _base_dict()
    del cfg_dict["lever_arms"]
    with pytest.raises(ConfigValidationError) as excinfo:
        build_config(cfg_dict)
    assert "lever_arms" in str(excinfo.value)


@pytest.mark.unit
def test_out_of_range_message_names_parameter_and_constraint():
    """An out-of-range value reports the dotted path and the violated bound."""
    cfg_dict = _base_dict()
    cfg_dict["quality_gates"]["max_longitudinal_accel_g"] = 5.0
    with pytest.raises(ConfigValidationError) as excinfo:
        build_config(cfg_dict)
    msg = str(excinfo.value)
    assert "quality_gates.max_longitudinal_accel_g" in msg
    assert "must be <= 2.0" in msg
    assert "got 5.0" in msg


@pytest.mark.unit
def test_wrong_type_message_names_parameter_and_type():
    """A wrong-typed value reports the dotted path and the expected type."""
    cfg_dict = _base_dict()
    cfg_dict["signals"]["gp_noise_variance"] = "not-a-number"
    with pytest.raises(ConfigValidationError) as excinfo:
        build_config(cfg_dict)
    msg = str(excinfo.value)
    assert "signals.gp_noise_variance" in msg
    assert "expected float, got str" in msg


@pytest.mark.unit
def test_unknown_estimator_rejected():
    """An unknown estimator name in estimators.enabled is rejected by name."""
    cfg_dict = _base_dict()
    cfg_dict["estimators"]["enabled"] = ["decel_knee", "totally_made_up"]
    with pytest.raises(ConfigValidationError) as excinfo:
        build_config(cfg_dict)
    msg = str(excinfo.value)
    assert "unknown estimator 'totally_made_up'" in msg
    assert "estimators.enabled" in msg


@pytest.mark.unit
def test_unknown_enum_choice_rejected():
    """An out-of-set enumerated choice is rejected, listing the allowed values."""
    cfg_dict = _base_dict()
    cfg_dict["geodesy"]["geoid_model"] = "MADE_UP_GEOID"
    with pytest.raises(ConfigValidationError) as excinfo:
        build_config(cfg_dict)
    msg = str(excinfo.value)
    assert "geodesy.geoid_model" in msg
    assert "must be one of" in msg


@pytest.mark.unit
def test_required_lever_arm_field_missing_rejected():
    """A lever-arm entry missing a required field errors (no sensible default)."""
    cfg_dict = _base_dict()
    del cfg_dict["lever_arms"]["B738"]["vertical_offset_m"]
    with pytest.raises(ConfigValidationError) as excinfo:
        build_config(cfg_dict)
    msg = str(excinfo.value)
    assert "lever_arms.B738.vertical_offset_m" in msg
    assert "required" in msg
