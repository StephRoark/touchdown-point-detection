"""Module: uncertainty quantification and calibration (Task 19).

Turns a fused touchdown estimate into calibrated, widened 90 % confidence
intervals for time (seconds) and along-runway distance (feet):

* :class:`ConformalCalibrator` -- split-conformal interval-width calibration so
  empirical coverage lands in the required 85-95 % band (Req 4.3, 4.4).
* gap / starvation / lever-arm widening factors (Req 9.2, 9.6, 7.5).
* :class:`UncertaintyQuantifier` -- orchestrates calibration + widening +
  low-confidence flagging (Req 4.1, 4.2, 4.5) into an :class:`UncertaintyResult`.
"""

from tdz.uncertainty.conformal import ConformalCalibrator, gaussian_multiplier
from tdz.uncertainty.quantifier import (
    FT_TO_M,
    M_TO_FT,
    UncertaintyQuantifier,
    UncertaintyResult,
)
from tdz.uncertainty.widening import (
    gap_widening_factor,
    missing_lever_arm_widening_factor,
    post_transition_starved,
    starvation_widening_factor,
)

__all__ = [
    "ConformalCalibrator",
    "gaussian_multiplier",
    "UncertaintyQuantifier",
    "UncertaintyResult",
    "FT_TO_M",
    "M_TO_FT",
    "gap_widening_factor",
    "post_transition_starved",
    "starvation_widening_factor",
    "missing_lever_arm_widening_factor",
]
