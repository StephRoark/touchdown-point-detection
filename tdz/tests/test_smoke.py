"""Smoke tests verifying the package skeleton imports and the runner works."""

import importlib

import pytest


@pytest.mark.unit
def test_top_level_package_imports():
    """The top-level tdz package imports and exposes a version."""
    import tdz

    assert tdz.__version__ == "0.1.0"


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name",
    [
        "tdz.io",
        "tdz.bracket",
        "tdz.timebase",
        "tdz.signals",
        "tdz.geo",
        "tdz.estimators",
        "tdz.estimators.physics",
        "tdz.estimators.changepoint",
        "tdz.estimators.learned",
        "tdz.fusion",
        "tdz.validation",
        "tdz.config",
    ],
)
def test_subpackages_import(module_name):
    """Every package in the pipeline tree imports cleanly."""
    assert importlib.import_module(module_name) is not None
