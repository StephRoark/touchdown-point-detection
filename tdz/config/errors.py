"""Configuration error types.

Validation is fail-fast: the first violation raises
:class:`ConfigValidationError` and no (partial) config object is returned. The
message always names the offending parameter as a dotted path and the
constraint that was violated, per design "Configuration Errors" and
Requirement 20.4 / Property 12.
"""

from __future__ import annotations


class ConfigError(Exception):
    """Base class for configuration problems (loading or validation)."""


class ConfigValidationError(ConfigError):
    """Raised when a configuration parameter fails schema validation.

    The message identifies the offending parameter (dotted path, e.g.
    ``quality_gates.max_longitudinal_accel_g``) and the violated constraint
    (e.g. ``must be <= 2.0, got 5.0``; ``expected float, got str``; or
    ``unknown estimator 'foo' in estimators.enabled``).
    """

    def __init__(self, path: str, constraint: str) -> None:
        self.path = path
        self.constraint = constraint
        super().__init__(f"{path}: {constraint}")
