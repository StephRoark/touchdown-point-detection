"""Touchdown point inference from ADS-B surveillance data.

The :mod:`tdz` package implements a batch-mode offline pipeline that estimates
sub-sample touchdown time from coarse ADS-B updates, maps it to an along-runway
position, and quantifies uncertainty.

Units convention: all internal computation is held in SI units (meters,
meters/second, seconds, radians). Conversion to presentation units (feet,
knots) is performed only at the output boundary.
"""

__version__ = "0.1.0"
