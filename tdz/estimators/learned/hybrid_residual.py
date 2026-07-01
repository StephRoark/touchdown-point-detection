"""Hybrid-residual touchdown estimator (Task 17, optional).

The hybrid-residual model keeps a **physical backbone** and trains the learned
model to predict the *correction* to that backbone rather than the absolute
touchdown time (design "Learned Estimators Detail -> Hybrid Residual (Optional)";
Req 5.3, 6.2). Concretely it composes two pieces:

* a **physics backbone** -- any
  :class:`~tdz.estimators.physics.base.PhysicsEstimator` (the deceleration-knee
  estimator by default). For each flight the backbone produces an *anchor*
  :class:`~tdz.models.TDEstimate` ``(t_td, sigma_t, diagnostics, ...)``; and
* a **learned residual model** -- three gradient-boosted-tree quantile
  regressors (median + a low/high pair) over the same fixed-length engineered
  window features the LightGBM estimator uses
  (:mod:`tdz.estimators.learned.features`). They predict the **residual**
  ``truth - anchor_t_td`` in seconds.

The final estimate reconstructs the absolute time from the physical backbone:

    t_td = physics_anchor_t_td + predicted_residual

Why residual-of-physics, not absolute time
------------------------------------------
Regressing the *residual* of a physics estimate (rather than the absolute epoch
time, or even an offset from the raw segmented knee) keeps a physically
meaningful backbone in charge: when the learned model abstains (predicts ~0
correction) the estimate degrades gracefully to the interpretable physics
anchor. The learned model's job is reduced to capturing the *systematic bias*
of the physics estimator -- a small, centred target that is far better posed
than absolute time, and one whose conditional distribution (modelled by the
quantile pair) already reflects the **total** error of the anchor (systematic +
random). Because the residual target ``truth - anchor`` already contains the
anchor's own scatter, ``sigma_t`` is taken from the residual interval alone and
the anchor ``sigma_t`` is **not** added again (that would double-count); the
anchor ``sigma_t`` is recorded in diagnostics for traceability.

Physics anchor always carried in the record (Req 6.2)
-----------------------------------------------------
When this learned estimator is the primary output, the physics anchor's
``t_td``, uncertainty, and diagnostic quantities must remain in the same output
record (Req 6.2). The anchor's full :class:`TDEstimate` is therefore copied into
this estimator's ``diagnostics`` under ``physics_anchor_*`` keys (and the anchor
``method_name``), so downstream assembly can surface the interpretable backbone
alongside the learned correction.

Uncertainty from the residual quantile pair
--------------------------------------------
As in the LightGBM estimator, the lower/upper residual quantiles are mapped to a
1-sigma width assuming approximate normality,

    sigma_t = (q_high - q_low) / (z_high - z_low),

with a :data:`MIN_SIGMA_T_S` floor, and the independently-fit boosters are
**quantile-rearranged** (sorted so ``low <= median <= high``) to guarantee the
point correction lies inside the reported interval. A repaired crossing -- or a
low-confidence physics anchor -- flags the estimate low-confidence.

On-ground upper bound (inherited)
---------------------------------
The estimator subclasses :class:`~tdz.estimators.physics.base.PhysicsEstimator`,
so the reconstructed ``t_td`` is run through the Requirement-18 on-ground-flag
upper bound by the base :meth:`estimate` exactly like every other estimator --
the learned correction cannot push the touchdown to at/after the on-ground
transition (Property 5). The backbone's own anchor is likewise bounded (the
backbone is itself a :class:`PhysicsEstimator`).

Unfit / unavailable behaviour
-----------------------------
Until :meth:`train` has been called the residual boosters do not exist, so
:meth:`_raw_estimate` returns a **failed** :class:`TDEstimate`
(``confidence="failed"``) with a ``model_not_trained`` diagnostic rather than
raising (matching the LightGBM/sequence estimators; no ``FailureReason`` enum
value denotes "untrained", so ``reason_code`` stays ``None``). When the physics
anchor itself fails for a flight, or the features cannot be built, the estimate
fails with the matching :class:`~tdz.models.FailureReason` (the anchor's reason
is carried through).

Reproducibility (Req 15.1 / 15.2)
---------------------------------
All three residual boosters are trained with a fixed ``seed`` and
CPU-deterministic LightGBM settings (``deterministic=True``,
``force_row_wise=True``, ``num_threads=1``); the categorical features use the
same deterministic hash as the LightGBM estimator. Two ``train`` calls with the
same seed, data, and backbone therefore produce identical predictions.

Units: SI throughout -- the residual target and ``sigma_t`` are seconds.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Final, Optional, Sequence

import numpy as np

from tdz.config.schema import SignalsConfig
from tdz.estimators.learned.features import (
    CATEGORICAL_FEATURE_INDICES,
    FEATURE_NAMES,
    N_FEATURES,
    WindowFeatures,
    extract_window_features,
)
from tdz.estimators.physics.base import (
    CONFIDENCE_FAILED,
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
    PhysicsEstimator,
    failed_estimate,
    make_estimate,
)
from tdz.estimators.physics.decel_knee import DecelKneeEstimator
from tdz.models import BaseEstimator, FailureReason, FlightRecord, QARTruthRecord, TDEstimate

__all__ = [
    "METHOD_NAME",
    "DEFAULT_QUANTILE_LOW",
    "DEFAULT_QUANTILE_HIGH",
    "DEFAULT_N_ESTIMATORS",
    "DEFAULT_NUM_LEAVES",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_MIN_CHILD_SAMPLES",
    "DEFAULT_SEED",
    "MIN_SIGMA_T_S",
    "MIN_TRAINING_SAMPLES",
    "HybridResidualEstimator",
]

#: Estimator identifier (matches the ``hybrid_residual`` id in ALLOWED_ESTIMATORS).
METHOD_NAME: Final[str] = "hybrid_residual"

#: Default lower/upper quantiles for the residual predictive interval (central
#: 90%). Overridable via the constructor.
DEFAULT_QUANTILE_LOW: Final[float] = 0.05
DEFAULT_QUANTILE_HIGH: Final[float] = 0.95

#: Default residual-booster hyperparameters (small/fast, unit-test scale; not a
#: production training run). Externalized as constructor params.
DEFAULT_N_ESTIMATORS: Final[int] = 200
DEFAULT_NUM_LEAVES: Final[int] = 15
DEFAULT_LEARNING_RATE: Final[float] = 0.05
DEFAULT_MIN_CHILD_SAMPLES: Final[int] = 5

#: Default master seed for the boosters (Req 15.2). Overridable per instance.
DEFAULT_SEED: Final[int] = 42

#: Absolute floor on the reported ``sigma_t`` (seconds); matches the other
#: estimators so fused uncertainties stay on a common scale.
MIN_SIGMA_T_S: Final[float] = 0.25

#: Minimum number of usable (features-buildable, anchor-usable, truth-matched)
#: training samples required to fit the residual boosters.
MIN_TRAINING_SAMPLES: Final[int] = 5


class HybridResidualEstimator(PhysicsEstimator):
    """Physics-backbone + learned-residual touchdown estimator (Req 5.3, 6.2).

    Parameters
    ----------
    physics_backbone:
        The physical backbone estimator (any
        :class:`~tdz.estimators.physics.base.PhysicsEstimator`). Defaults to a
        :class:`~tdz.estimators.physics.decel_knee.DecelKneeEstimator` -- the
        velocity-stream knee anchor that runs without geometric altitude.
    quantile_low, quantile_high:
        Lower/upper quantiles for the residual predictive interval (0.05/0.95).
    n_estimators, num_leaves, learning_rate, min_child_samples:
        LightGBM hyperparameters for the residual boosters.
    seed:
        Master random seed propagated to every booster (Req 15.2).
    signals_config:
        :class:`~tdz.config.schema.SignalsConfig` for the feature derivative
        channels; defaults to the feature module's default.
    n_segments:
        Segments for the segmented-fit feature reference (2 or 3).
    min_sigma_t_s:
        Floor on the reported ``sigma_t`` (seconds).
    """

    method_name = METHOD_NAME

    def __init__(
        self,
        physics_backbone: Optional[BaseEstimator] = None,
        *,
        quantile_low: float = DEFAULT_QUANTILE_LOW,
        quantile_high: float = DEFAULT_QUANTILE_HIGH,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        num_leaves: int = DEFAULT_NUM_LEAVES,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        min_child_samples: int = DEFAULT_MIN_CHILD_SAMPLES,
        seed: int = DEFAULT_SEED,
        signals_config: Optional[SignalsConfig] = None,
        n_segments: int = 2,
        min_sigma_t_s: float = MIN_SIGMA_T_S,
    ) -> None:
        if not (0.0 < quantile_low < 0.5 < quantile_high < 1.0):
            raise ValueError(
                "require 0 < quantile_low < 0.5 < quantile_high < 1, got "
                f"({quantile_low}, {quantile_high})"
            )
        self.physics_backbone: BaseEstimator = (
            physics_backbone if physics_backbone is not None else DecelKneeEstimator()
        )
        self.quantile_low = float(quantile_low)
        self.quantile_high = float(quantile_high)
        self.n_estimators = int(n_estimators)
        self.num_leaves = int(num_leaves)
        self.learning_rate = float(learning_rate)
        self.min_child_samples = int(min_child_samples)
        self.seed = int(seed)
        self.signals_config = signals_config
        self.n_segments = int(n_segments)
        self.min_sigma_t_s = float(min_sigma_t_s)

        # z-span of the configured quantile pair under normality (for sigma_t).
        nd = NormalDist()
        self._z_span = nd.inv_cdf(self.quantile_high) - nd.inv_cdf(self.quantile_low)

        # Residual boosters (None until trained).
        self._model_median = None
        self._model_low = None
        self._model_high = None
        self._n_train_samples: int = 0
        self._n_train_skipped: int = 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """``True`` once :meth:`train` has fitted the three residual boosters."""
        return self._model_median is not None

    # ------------------------------------------------------------------
    # Backbone / feature helpers
    # ------------------------------------------------------------------

    def _physics_anchor(self, flight: FlightRecord) -> TDEstimate:
        """Run the physics backbone to produce the anchor estimate."""
        return self.physics_backbone.estimate(flight)

    def _build_feature_vector(
        self, flight: FlightRecord
    ) -> tuple[Optional[WindowFeatures], Optional[FailureReason]]:
        """Extract the fixed-length window-feature vector for one flight."""
        return extract_window_features(
            flight,
            self.signals_config,
            n_segments=self.n_segments,
        )

    @staticmethod
    def _anchor_usable(anchor: TDEstimate) -> bool:
        """True when the anchor did not fail and carries a finite ``t_td``."""
        return (
            anchor.confidence != CONFIDENCE_FAILED
            and anchor.t_td is not None
            and np.isfinite(anchor.t_td)
        )

    def build_training_matrix(
        self,
        flights: Sequence[FlightRecord],
        truths: Sequence[QARTruthRecord],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build ``(X, y, skipped_flight_ids)`` from flights + matching truths.

        Flights are matched to truths by ``flight_id``. The target ``y`` for each
        usable flight is the physics **residual** ``touchdown_time_qar -
        anchor_t_td``, where the anchor is the physics backbone's estimate (the
        QAR truth is used ONLY here, to form the target -- never as a feature).
        Flights whose features cannot be built, whose physics anchor failed, or
        that have no matching truth, are skipped and their ids returned.
        """
        truth_by_id: dict[str, QARTruthRecord] = {t.flight_id: t for t in truths}
        rows: list[np.ndarray] = []
        targets: list[float] = []
        skipped: list[str] = []

        for flight in flights:
            truth = truth_by_id.get(flight.flight_id)
            if truth is None:
                skipped.append(flight.flight_id)
                continue
            features, _reason = self._build_feature_vector(flight)
            if features is None:
                skipped.append(flight.flight_id)
                continue
            anchor = self._physics_anchor(flight)
            if not self._anchor_usable(anchor):
                skipped.append(flight.flight_id)
                continue
            residual = float(truth.touchdown_time_qar) - float(anchor.t_td)
            if not np.isfinite(residual):
                skipped.append(flight.flight_id)
                continue
            rows.append(features.values)
            targets.append(residual)

        if rows:
            x = np.vstack(rows)
        else:
            x = np.empty((0, N_FEATURES), dtype=float)
        y = np.asarray(targets, dtype=float)
        return x, y, skipped

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        flights: Sequence[FlightRecord],
        truths: Sequence[QARTruthRecord],
    ) -> "HybridResidualEstimator":
        """Fit the three residual quantile boosters on labeled landings; return ``self``.

        Raises
        ------
        ImportError
            If LightGBM is not installed.
        ValueError
            If fewer than :data:`MIN_TRAINING_SAMPLES` usable samples remain
            after feature extraction / anchor evaluation / truth matching.
        """
        import lightgbm as lgb  # local import: optional heavy dependency

        x, y, skipped = self.build_training_matrix(flights, truths)
        if x.shape[0] < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"need at least {MIN_TRAINING_SAMPLES} usable training samples, "
                f"got {x.shape[0]} (skipped {len(skipped)})"
            )

        self._n_train_samples = int(x.shape[0])
        self._n_train_skipped = len(skipped)

        self._model_median = self._fit_quantile_model(lgb, x, y, alpha=0.5)
        self._model_low = self._fit_quantile_model(lgb, x, y, alpha=self.quantile_low)
        self._model_high = self._fit_quantile_model(lgb, x, y, alpha=self.quantile_high)
        return self

    #: ``fit`` is an alias for :meth:`train` (sklearn-style naming).
    fit = train

    def _fit_quantile_model(self, lgb, x: np.ndarray, y: np.ndarray, *, alpha: float):
        """Fit one quantile LightGBM regressor with deterministic settings."""
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            random_state=self.seed,
            n_jobs=1,
            num_threads=1,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        # Bare numpy arrays are used at fit and predict time (fixed column order);
        # feature names are bound by position to FEATURE_NAMES, matching the
        # LightGBM window-feature estimator.
        model.fit(
            x,
            y,
            categorical_feature=list(CATEGORICAL_FEATURE_INDICES),
        )
        return model

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _raw_estimate(self, flight: FlightRecord) -> TDEstimate:
        """Predict ``t_td = anchor + residual`` for one flight; see module docstring."""
        if not self.is_trained:
            # No booster: fail gracefully (no enum value denotes "untrained";
            # the detail is carried in diagnostics, reason_code stays None).
            return make_estimate(
                t_td=float("nan"),
                sigma_t=float("inf"),
                confidence="failed",
                method_name=self.method_name,
                diagnostics={"detail": "model_not_trained"},
                reason=None,
            )

        # Physical backbone first: it is the anchor the residual corrects, and
        # must be present in the record (Req 6.2) even on the failure paths.
        anchor = self._physics_anchor(flight)
        anchor_diag = self._anchor_diagnostics(anchor)

        if not self._anchor_usable(anchor):
            # The backbone could not place a touchdown -> the hybrid cannot
            # correct a non-existent anchor. Fail, carrying the anchor's reason
            # and its diagnostics for traceability.
            reason_value = anchor.diagnostics.get("reason_code")
            reason = self._reason_from_value(reason_value)
            return make_estimate(
                t_td=float("nan"),
                sigma_t=float("inf"),
                confidence=CONFIDENCE_FAILED,
                method_name=self.method_name,
                diagnostics={**anchor_diag, "detail": "physics_anchor_failed"},
                reason=reason,
            )

        features, reason = self._build_feature_vector(flight)
        if features is None:
            return make_estimate(
                t_td=float("nan"),
                sigma_t=float("inf"),
                confidence=CONFIDENCE_FAILED,
                method_name=self.method_name,
                diagnostics=anchor_diag,
                reason=reason or FailureReason.INSUFFICIENT_SAMPLES,
            )

        x = features.values.reshape(1, -1)
        # Predict via the underlying boosters (``booster_``) to avoid the sklearn
        # feature-name validation re-run on every call (noisy for bare arrays).
        raw_median = float(self._model_median.booster_.predict(x)[0])
        raw_low = float(self._model_low.booster_.predict(x)[0])
        raw_high = float(self._model_high.booster_.predict(x)[0])

        # Quantile rearrangement (Chernozhukov et al.): sort the trio so the
        # monotonicity ``low <= median <= high`` holds even when the
        # independently-fit boosters cross on small data. Guarantees the point
        # correction lies inside the reported residual interval.
        ordered = sorted((raw_low, raw_median, raw_high))
        residual_low, residual_median, residual_high = ordered
        crossed = not (raw_low <= raw_median <= raw_high)

        anchor_t_td = float(anchor.t_td)
        t_td = anchor_t_td + residual_median
        t_low = anchor_t_td + residual_low
        t_high = anchor_t_td + residual_high

        sigma_t = self._sigma_from_quantiles(residual_low, residual_high)

        # Low-confidence when the residual interval was repaired OR the physics
        # backbone itself was low-confidence (a shaky anchor begets a shaky
        # correction).
        anchor_low = anchor.confidence == CONFIDENCE_LOW
        if crossed:
            confidence = CONFIDENCE_LOW
            reason_code = FailureReason.WIDE_CONFIDENCE_INTERVAL
        elif anchor_low:
            confidence = CONFIDENCE_LOW
            reason_code = self._reason_from_value(
                anchor.diagnostics.get("reason_code")
            ) or FailureReason.ESTIMATOR_DISAGREEMENT
        else:
            confidence = CONFIDENCE_NORMAL
            reason_code = None

        diagnostics = {
            **anchor_diag,
            "predicted_residual_s": residual_median,
            "residual_low_s": residual_low,
            "residual_high_s": residual_high,
            "quantile_low": self.quantile_low,
            "quantile_high": self.quantile_high,
            "z_span": self._z_span,
            "t_td_ci_lower": t_low,
            "t_td_ci_upper": t_high,
            "quantile_crossing_repaired": crossed,
            "reference_time": features.reference_time,
            "reference_kind": features.reference_kind,
            "n_train_samples": self._n_train_samples,
        }
        return make_estimate(
            t_td=t_td,
            sigma_t=sigma_t,
            confidence=confidence,
            method_name=self.method_name,
            diagnostics=diagnostics,
            reason=reason_code,
        )

    def _anchor_diagnostics(self, anchor: TDEstimate) -> dict:
        """Copy the physics anchor's t_td/uncertainty/diagnostics into the record (Req 6.2)."""
        return {
            "physics_anchor_method": anchor.method_name,
            "physics_anchor_t_td": (
                float(anchor.t_td) if anchor.t_td is not None else None
            ),
            "physics_anchor_sigma_t": (
                float(anchor.sigma_t) if anchor.sigma_t is not None else None
            ),
            "physics_anchor_confidence": anchor.confidence,
            "physics_anchor_diagnostics": dict(anchor.diagnostics),
        }

    @staticmethod
    def _reason_from_value(reason_value: Optional[str]) -> Optional[FailureReason]:
        """Map a diagnostics ``reason_code`` string back to its :class:`FailureReason`."""
        if reason_value is None:
            return None
        try:
            return FailureReason(reason_value)
        except ValueError:  # pragma: no cover - unknown/foreign reason string
            return None

    def _sigma_from_quantiles(self, residual_low: float, residual_high: float) -> float:
        """Map the (sorted) residual quantile spread to a 1-sigma width (see module docstring)."""
        width = max(residual_high - residual_low, 0.0)
        if self._z_span > 0.0:
            sigma = width / self._z_span
        else:  # pragma: no cover - guarded by constructor validation
            sigma = width
        return max(float(sigma), self.min_sigma_t_s)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def feature_importances(self, importance_type: str = "gain") -> dict[str, float]:
        """Return ``{feature_name: importance}`` from the median residual booster.

        Importances support the safety narrative (design). Returns an empty dict
        when the estimator has not been trained.

        Parameters
        ----------
        importance_type:
            ``"gain"`` (default; total split gain) or ``"split"`` (split counts).
        """
        if not self.is_trained:
            return {}
        booster = self._model_median.booster_
        importances = booster.feature_importance(importance_type=importance_type)
        return {name: float(v) for name, v in zip(FEATURE_NAMES, importances)}
