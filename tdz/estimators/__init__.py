"""Module 4: Estimators.

All estimators emit the common contract ``(t_td, sigma_t, diagnostics)``, making
the fusion and mapping layers method-agnostic. Subpackages group the estimator
families: physics, change-point, and learned.
"""
