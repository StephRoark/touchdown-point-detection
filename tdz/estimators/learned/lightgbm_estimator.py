"""LightGBM window-feature touchdown estimator (Task 15).

The first **learned** estimator (Req 5.3). It reduces each landing to a
fixed-length engineered feature vector (:mod:`tdz.estimators.learned.features`)
and feeds it to gradient-boosted trees that predict a touchdown-time **offset**
relative to the physics-knee reference, plus a quantile pair for uncertainty
(design "Learned Estimators Detail -> LightGBM Window Features").

What is predicted, and against what reference
---------------------------------------------
The boosters predict the **offset** ``t_td - reference`` in seconds, where
``reference`` is the segmented-fit groundspeed breakpoint (the deceleration knee
-- see :mod:`tdz.estimators.learned.features`). Predicting a small, centred
offset rather than an absolute epoch time makes the target well-posed and lets
the model *refine* the physics knee instead of relearning wall-clock time. At
predict time the absolute touchdown is reconstructed as
``t_td = reference + offset_median``.

Three boosters -> a predictive interval -> ``sigma_t``
------------------------------------------------------
Uncertainty is produced by **quantile regression**: three LightGBM models share
the features but differ only in the quantile they target --

* a **median** model (``alpha = 0.5``) gives the point offset, and
* a **lower / upper** pair (default ``alpha = 0.05`` / ``0.95``) gives a central
  predictive interval ``[q_low, q_high]``.

The interval is mapped to a 1-sigma width assuming approximate normality:

    sigma_t = (q_high - q_low) / (z_high - z_low)

where ``z_x = Phi^{-1}(alpha_x)`` is the standard-normal quantile. For the
default 5/95 pair ``z_high - z_low = 2 * 1.6449 = 3.2897``, so ``sigma_t`` is the
interval width divided by ~3.29. A :data:`MIN_SIGMA_T_S` floor keeps it on the
same scale as the physics/change-point estimators. The three boosters are fit
independently, so on small data their predictions can **cross** (the 0.5 median
falling outside ``[q_low, q_high]``); this is repaired by **quantile
rearrangement** -- sorting the trio so ``low <= median <= high`` (a standard,
monotone correction, Chernozhukov et al.) -- which guarantees the point estimate
lies inside the reported interval. A repaired crossing is flagged low-confidence.

On-ground upper bound (inherited)
---------------------------------
The estimator subclasses :class:`~tdz.estimators.physics.base.PhysicsEstimator`,
so the reconstructed ``t_td`` is run through the Requirement-18 on-ground-flag
upper bound by the base :meth:`estimate` exactly like every other estimator --
the learned model cannot output a touchdown at or after the on-ground transition
(Property 5).

Unfit / unavailable behaviour
-----------------------------
Until :meth:`train` has been called the estimator has no boosters, so
:meth:`_raw_estimate` returns a **failed** :class:`~tdz.models.TDEstimate`
(``confidence="failed"``) with a ``model_not_trained`` diagnostic rather than
raising. (No ``FailureReason`` enum value cleanly denotes "model not trained";
per the task guidance no enum value is invented -- ``reason_code`` is left
``None`` and the detail is carried in diagnostics.) When the features cannot be
built (no groundspeed / too few samples) it fails with the matching
:class:`~tdz.models.FailureReason` from feature extraction.

Reproducibility (Req 15.1 / 15.2)
---------------------------------
All three boosters are trained with a fixed ``seed`` and CPU-deterministic
LightGBM settings (``deterministic=True``, ``force_row_wise=True``,
``num_threads=1``). Two ``train`` calls with the same seed and data therefore
produce bit-identical boosters and hence identical predictions. The categorical
features use a deterministic hash (not Python's salted ``hash``) for the same
reason.

Units: SI throughout -- the offset target and ``sigma_t`` are seconds.
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
    CONFIDENCE_LOW,
    CONFIDENCE_NORMAL,
    PhysicsEstimator,
    failed_estimate,
    make_estimate,
)
from tdz.models import FailureReason, FlightRecord, QARTruthRecord, TDEstimate

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
    "LightGbmTouchdownEstimator",
]

#: Estimator identifier (matches the ``lightgbm`` id in ``ALLOWED_ESTIMATORS``).
METHOD_NAME: Final[str] = "lightgbm"

#: Default lower/upper quantiles for the predictive interval (a central 90%
#: interval). Externalizable via the constructor.
DEFAULT_QUANTILE_LOW: Final[float] = 0.05
DEFAULT_QUANTILE_HIGH: Final[float] = 0.95

#: Default booster hyperparameters. Kept small/fast (this is a unit-test-scale
#: model, not a production training run) and externalized as constructor params.
DEFAULT_N_ESTIMATORS: Final[int] = 200
DEFAULT_NUM_LEAVES: Final[int] = 15
DEFAULT_LEARNING_RATE: Final[float] = 0.05
DEFAULT_MIN_CHILD_SAMPLES: Final[int] = 5

#: Default master seed for the boosters (Req 15.2). Overridable per instance.
DEFAULT_SEED: Final[int] = 42

#: Absolute floor on the reported ``sigma_t`` (seconds); matches the physics /
#: change-point estimators so fused uncertainties are on a common scale.
MIN_SIGMA_T_S: Final[float] = 0.25

#: Minimum number of usable (features-buildable, truth-matched) training samples
#: required to fit the boosters.
MIN_TRAINING_SAMPLES: Final[int] = 5


class LightGbmTouchdownEstimator(PhysicsEstimator):
    """Gradient-boosted-tree window-feature touchdown estimator (Req 5.3).

    Parameters
    ----------
    quantile_low, quantile_high:
        Lower/upper quantiles for the predictive interval (defaults 0.05/0.95).
    n_estimators, num_leaves, learning_rate, min_child_samples:
        LightGBM hyperparameters (small defaults for fast tests; see constants).
    seed:
        Master random seed propagated to every booster (Req 15.2).
    signals_config:
        :class:`~tdz.config.schema.SignalsConfig` for the derivative channels the
        features summarise; defaults to the feature module's default.
    n_segments:
        Segments for the segmented-fit reference (2 or 3).
    min_sigma_t_s:
        Floor on the reported ``sigma_t`` (seconds).
    """

    method_name = METHOD_NAME

    def __init__(
        self,
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

        # Boosters (None until trained).
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
        """``True`` once :meth:`train` has fitted the three boosters."""
        return self._model_median is not None

    # ------------------------------------------------------------------
    # Feature matrix construction
    # ------------------------------------------------------------------

    def _build_feature_vector(
        self, flight: FlightRecord
    ) -> tuple[Optional[WindowFeatures], Optional[FailureReason]]:
        """Extract the window-feature vector for one flight (delegates to features)."""
        return extract_window_features(
            flight,
            self.signals_config,
            n_segments=self.n_segments,
        )

    def build_training_matrix(
        self,
        flights: Sequence[FlightRecord],
        truths: Sequence[QARTruthRecord],
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Build ``(X, y, skipped_flight_ids)`` from flights + matching truths.

        Flights are matched to truths by ``flight_id``. The target ``y`` for each
        usable flight is the **offset** ``touchdown_time_qar - reference`` (the
        QAR truth is used ONLY here, to form the target -- never as a feature).
        Flights whose features cannot be built, or that have no matching truth,
        are skipped and their ids returned.
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
            offset = float(truth.touchdown_time_qar) - features.reference_time
            if not np.isfinite(offset):
                skipped.append(flight.flight_id)
                continue
            rows.append(features.values)
            targets.append(offset)

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
    ) -> "LightGbmTouchdownEstimator":
        """Fit the three quantile boosters on labeled landings; return ``self``.

        Raises
        ------
        ImportError
            If LightGBM is not installed.
        ValueError
            If fewer than :data:`MIN_TRAINING_SAMPLES` usable samples remain
            after feature extraction / truth matching.
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
        # ``feature_name`` is intentionally NOT passed: the boosters are trained
        # and queried with bare numpy arrays, so binding names at fit time only
        # triggers sklearn's "X has no feature names" warning at predict time.
        # Importances are mapped back to FEATURE_NAMES by column position (the
        # column order is fixed by the feature module), which is equivalent.
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
        """Predict ``t_td`` (offset + quantiles) for one flight; see module docstring."""
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

        features, reason = self._build_feature_vector(flight)
        if features is None:
            return failed_estimate(
                self.method_name,
                reason or FailureReason.INSUFFICIENT_SAMPLES,
            )

        x = features.values.reshape(1, -1)
        # Predict via the underlying boosters (``booster_``) rather than the
        # sklearn wrapper: the wrapper re-runs sklearn feature-name validation on
        # every call (noisy for the bare numpy arrays used here), while the
        # booster consumes the fixed-column-order matrix directly.
        raw_median = float(self._model_median.booster_.predict(x)[0])
        raw_low = float(self._model_low.booster_.predict(x)[0])
        raw_high = float(self._model_high.booster_.predict(x)[0])

        # Quantile rearrangement (Chernozhukov et al.): the three boosters are fit
        # independently, so on small data their predictions can cross (e.g. the
        # 0.5 median falling outside [q05, q95]). Sorting the trio restores the
        # required monotonicity ``low <= median <= high`` -- a standard, monotone
        # correction -- which guarantees the point estimate lies within the
        # reconstructed interval. A crossing is recorded and flagged low-confidence.
        ordered = sorted((raw_low, raw_median, raw_high))
        offset_low, offset_median, offset_high = ordered
        crossed = not (raw_low <= raw_median <= raw_high)

        reference = features.reference_time
        t_td = reference + offset_median
        t_low = reference + offset_low
        t_high = reference + offset_high

        sigma_t = self._sigma_from_quantiles(offset_low, offset_high)

        confidence = CONFIDENCE_LOW if crossed else CONFIDENCE_NORMAL
        reason_code = FailureReason.WIDE_CONFIDENCE_INTERVAL if crossed else None

        diagnostics = {
            "reference_time": reference,
            "reference_kind": features.reference_kind,
            "offset_median_s": offset_median,
            "offset_low_s": offset_low,
            "offset_high_s": offset_high,
            "quantile_low": self.quantile_low,
            "quantile_high": self.quantile_high,
            "z_span": self._z_span,
            "t_td_ci_lower": t_low,
            "t_td_ci_upper": t_high,
            "quantile_crossing_repaired": crossed,
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

    def _sigma_from_quantiles(self, offset_low: float, offset_high: float) -> float:
        """Map the (sorted) quantile spread to a 1-sigma width (see module docstring)."""
        width = max(offset_high - offset_low, 0.0)
        if self._z_span > 0.0:
            sigma = width / self._z_span
        else:  # pragma: no cover - guarded by constructor validation
            sigma = width
        return max(float(sigma), self.min_sigma_t_s)

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def feature_importances(self, importance_type: str = "gain") -> dict[str, float]:
        """Return ``{feature_name: importance}`` from the median booster.

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
