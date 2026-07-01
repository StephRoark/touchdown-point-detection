"""TCN/BiLSTM sequence-model touchdown estimator (Task 16).

The most expressive **learned** estimator (Req 5.3, design "Learned Estimators
Detail -> TCN/BiLSTM Sequence Model"). Unlike the LightGBM model
(:mod:`tdz.estimators.learned.lightgbm_estimator`), which reduces a landing to a
fixed-length engineered vector, this estimator consumes the **full per-sample
sequence** of physics-derived channels and predicts a touchdown *distribution*
over the landing window.

What it consumes
----------------
Per-timestep channels are taken from the Task-11 builder
(:func:`tdz.signals.features.build_feature_channels`) -- NOT the window-feature
reduction -- assembled on the **velocity timebase** (the densest, most directly
touchdown-relevant stream). The position-timebase channels (distance-to-threshold,
height-above-runway) are interpolated onto the velocity timebase, with a binary
*availability* flag so the model can tell a genuinely-zero value from a missing
one (FR24 velocity-only sources have no height channel). The explicit
**time-delta** channel makes the model aware of the irregular 4-5 s cadence, and
a **relative-time** channel (sample time minus the physics-knee reference) gives
it a physical sense of where the deceleration knee sits. Static context --
aircraft type and ADS-B source -- enters as learned **embeddings** broadcast
across every timestep.

Soft Gaussian labels -> a distribution, not a point
---------------------------------------------------
Hard one-hot "the touchdown is at sample k" labels are sparse and brittle to
label-time clock noise (design key decision "Soft Gaussian labels for sequence
model"). Instead each training sequence's target is a **Gaussian bump** centred
on the QAR touchdown time with width :data:`DEFAULT_LABEL_SIGMA_S`, renormalised
to a proper probability distribution over the window's timesteps. The model
outputs a per-timestep ``P(touchdown)`` (a softmax over time), trained to match
the soft-label distribution by cross-entropy (equivalently KL up to a constant).

Output: expected value -> ``t_td``, distribution width -> uncertainty
---------------------------------------------------------------------
At predict time the per-timestep distribution ``p_i`` is collapsed to a
sub-sample estimate by its **expected value** over the sample times,

    t_td = sum_i p_i * t_i,

and its **width** gives the uncertainty,

    sigma_t = sqrt( sum_i p_i * (t_i - t_td)^2 ),

so a sharp, confident distribution yields a small ``sigma_t`` and a smeared one a
large ``sigma_t`` -- the uncertainty falls straight out of the predicted shape
rather than a separate head. A :data:`MIN_SIGMA_T_S` floor keeps it on the same
scale as the physics/change-point estimators.

Optional deep ensemble (epistemic uncertainty)
----------------------------------------------
With ``n_ensemble > 1`` the estimator trains an ensemble of independently
seeded models (design "Optional: Deep ensemble (~5 models)"). Their per-timestep
distributions are averaged, and the spread of their individual expected values
is added in quadrature as an **epistemic** term -- so disagreement between models
widens the reported uncertainty honestly.

On-ground upper bound (inherited)
---------------------------------
The estimator subclasses :class:`~tdz.estimators.physics.base.PhysicsEstimator`,
so the expected-value ``t_td`` is run through the Requirement-18 on-ground-flag
upper bound by the base :meth:`estimate` exactly like every other estimator --
the learned model cannot output a touchdown at or after the on-ground transition
(Property 5).

Reproducibility (Req 15.1 / 15.2)
---------------------------------
Neural training is seeded from a single master ``seed`` (each ensemble member
from ``seed + k``); on CPU with full-batch gradient descent this is bit-identical
across runs. ``deterministic=True`` (the default) additionally requests
PyTorch's deterministic algorithms; the chosen mode is recorded in the estimate
diagnostics (``deterministic_mode``) per Req 15.1.

Units: SI throughout -- channels in m/s, m/s^2, m/s^3, m, s; ``t_td`` and
``sigma_t`` are seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Optional, Sequence

import numpy as np

from tdz.config.schema import SignalsConfig
from tdz.estimators.learned.features import (
    AIRCRAFT_TYPE_VOCAB_SIZE,
    encode_aircraft_type,
    encode_source,
)
from tdz.estimators.physics.base import (
    CONFIDENCE_NORMAL,
    PhysicsEstimator,
    failed_estimate,
    make_estimate,
)
from tdz.models import FailureReason, FlightRecord, QARTruthRecord, TDEstimate
from tdz.signals.features import build_feature_channels

__all__ = [
    "METHOD_NAME",
    "DEFAULT_LABEL_SIGMA_S",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_AIRCRAFT_EMBED_DIM",
    "DEFAULT_SOURCE_EMBED_DIM",
    "DEFAULT_KERNEL_SIZE",
    "DEFAULT_DILATIONS",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_N_ENSEMBLE",
    "DEFAULT_SEED",
    "MIN_SIGMA_T_S",
    "MIN_SEQUENCE_SAMPLES",
    "MIN_TRAINING_SAMPLES",
    "N_SOURCE_CODES",
    "CONTINUOUS_CHANNEL_NAMES",
    "FLAG_CHANNEL_NAMES",
    "SequenceInput",
    "SequenceModelEstimator",
]

#: Estimator identifier (matches the ``sequence_model`` id in ALLOWED_ESTIMATORS).
METHOD_NAME: Final[str] = "sequence_model"

#: Soft-label Gaussian width (seconds). Wide enough that adjacent 4-5 s samples
#: share label mass (so the target is not one-hot) but narrow enough to localise
#: the touchdown. Overridable per instance.
DEFAULT_LABEL_SIGMA_S: Final[float] = 2.5

#: Network hyperparameters (small by default: this is a unit-test-scale model,
#: not a production training run). All overridable via the constructor.
DEFAULT_HIDDEN_DIM: Final[int] = 32
DEFAULT_AIRCRAFT_EMBED_DIM: Final[int] = 4
DEFAULT_SOURCE_EMBED_DIM: Final[int] = 2
DEFAULT_KERNEL_SIZE: Final[int] = 3
DEFAULT_DILATIONS: Final[tuple[int, ...]] = (1, 2, 4)
DEFAULT_N_EPOCHS: Final[int] = 120
DEFAULT_LEARNING_RATE: Final[float] = 0.02

#: Number of models in the deep ensemble (1 = single model). ~5 is the design
#: suggestion for epistemic uncertainty.
DEFAULT_N_ENSEMBLE: Final[int] = 1

#: Default master seed (Req 15.2). Overridable per instance.
DEFAULT_SEED: Final[int] = 42

#: Floor on the reported ``sigma_t`` (seconds); matches the other estimators so
#: fused uncertainties share a scale.
MIN_SIGMA_T_S: Final[float] = 0.25

#: Minimum finite groundspeed samples needed to form a sequence.
MIN_SEQUENCE_SAMPLES: Final[int] = 4

#: Minimum usable (sequence-buildable, truth-matched) training sequences.
MIN_TRAINING_SAMPLES: Final[int] = 5

#: Distinct ADS-B source codes (see ``encode_source``: aireon=0, fr24=1, other=2).
N_SOURCE_CODES: Final[int] = 3

#: Standardised continuous per-timestep channels, in column order.
CONTINUOUS_CHANNEL_NAMES: Final[tuple[str, ...]] = (
    "groundspeed_mps",
    "deceleration_mps2",
    "jerk_mps3",
    "derivative_uncertainty",
    "time_delta_s",
    "distance_to_threshold_m",
    "height_above_runway_m",
    "rel_time_s",
)

#: Binary availability flags (NOT standardised), in column order. They let the
#: model distinguish a true zero from an imputed-missing value.
FLAG_CHANNEL_NAMES: Final[tuple[str, ...]] = (
    "distance_available",
    "height_available",
)

_N_CONTINUOUS: Final[int] = len(CONTINUOUS_CHANNEL_NAMES)
_N_FLAGS: Final[int] = len(FLAG_CHANNEL_NAMES)


# ---------------------------------------------------------------------------
# Sequence input assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceInput:
    """Per-flight model input assembled on the velocity timebase (SI units).

    Attributes
    ----------
    times:
        Sample times (epoch seconds), length ``T`` -- the axis over which the
        predicted distribution's expectation and width are taken.
    continuous:
        ``(T, N_CONTINUOUS)`` raw (un-standardised) continuous channels; may
        contain ``NaN`` for unavailable position-timebase channels.
    flags:
        ``(T, N_FLAGS)`` binary availability flags (never standardised).
    aircraft_index:
        Hashed aircraft-type code (embedding index).
    source_index:
        ADS-B source code (embedding index).
    reference_time:
        The physics-knee reference (segmented breakpoint) used for ``rel_time``.
    """

    times: np.ndarray
    continuous: np.ndarray
    flags: np.ndarray
    aircraft_index: int
    source_index: int
    reference_time: float


def _interp_to(
    src_times: np.ndarray, src_values: np.ndarray, dst_times: np.ndarray
) -> tuple[np.ndarray, bool]:
    """Interpolate a (possibly NaN-laden) channel onto ``dst_times``.

    Returns ``(values, available)``. ``available`` is ``False`` (and values are
    all-NaN) when fewer than two finite source samples exist -- e.g. the height
    channel on a velocity-only FR24 record.
    """
    s_t = np.asarray(src_times, dtype=float)
    s_v = np.asarray(src_values, dtype=float)
    finite = np.isfinite(s_t) & np.isfinite(s_v)
    if int(np.count_nonzero(finite)) < 2:
        return np.full(np.asarray(dst_times).shape, np.nan), False
    order = np.argsort(s_t[finite])
    xt = s_t[finite][order]
    xv = s_v[finite][order]
    # np.interp clamps outside the support to the end values, which is the
    # desired behaviour here (hold the nearest known distance/height).
    return np.interp(np.asarray(dst_times, dtype=float), xt, xv), True


def build_sequence_input(
    flight: FlightRecord,
    config: Optional[SignalsConfig] = None,
    *,
    breakpoint_time: Optional[float] = None,
) -> tuple[Optional[SequenceInput], Optional[FailureReason]]:
    """Assemble the per-timestep model input for one flight.

    Returns ``(input, None)`` on success or ``(None, reason)`` when the landing
    cannot support a sequence:

    * :attr:`FailureReason.NO_GROUNDSPEED` -- groundspeed missing entirely.
    * :attr:`FailureReason.INSUFFICIENT_SAMPLES` -- fewer than
      :data:`MIN_SEQUENCE_SAMPLES` finite groundspeed samples.

    The channels are taken from :func:`tdz.signals.features.build_feature_channels`
    (the per-sample builder), assembled on the velocity timebase; the
    position-timebase distance/height channels are interpolated onto it.
    """
    gs = np.asarray(flight.groundspeeds, dtype=float)
    if gs.size == 0 or np.all(np.isnan(gs)):
        return None, FailureReason.NO_GROUNDSPEED

    channels = build_feature_channels(flight, config, breakpoint_time=breakpoint_time)

    v_times = np.asarray(channels.velocity_times, dtype=float)
    gs_mps = np.asarray(channels.groundspeed_mps, dtype=float)
    finite = np.isfinite(v_times) & np.isfinite(gs_mps)
    n_finite = int(np.count_nonzero(finite))
    if n_finite < MIN_SEQUENCE_SAMPLES:
        return None, FailureReason.INSUFFICIENT_SAMPLES

    idx = np.argsort(v_times[finite])
    times = v_times[finite][idx]

    def _sel(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=float)
        if a.shape != v_times.shape:
            # Channel not on the velocity timebase / unexpected length: NaN-fill.
            return np.full(times.shape, np.nan)
        return a[finite][idx]

    groundspeed = _sel(channels.groundspeed_mps)
    decel = _sel(channels.deceleration_mps2)
    jerk = _sel(channels.jerk_mps3)
    deriv_unc = _sel(channels.derivative_uncertainty)
    time_delta = _sel(channels.velocity_time_deltas)

    distance, dist_avail = _interp_to(
        channels.position_times, channels.distance_to_threshold_m, times
    )
    height, height_avail = _interp_to(
        channels.position_times, channels.height_above_runway_m, times
    )

    reference = channels.segmented_breakpoint_time
    if reference is None or not np.isfinite(reference):
        reference = float(np.mean(times))
    rel_time = times - float(reference)

    continuous = np.column_stack(
        [
            groundspeed,
            decel,
            jerk,
            deriv_unc,
            time_delta,
            distance,
            height,
            rel_time,
        ]
    ).astype(float)
    flags = np.column_stack(
        [
            np.full(times.shape, 1.0 if dist_avail else 0.0),
            np.full(times.shape, 1.0 if height_avail else 0.0),
        ]
    ).astype(float)

    return (
        SequenceInput(
            times=times,
            continuous=continuous,
            flags=flags,
            aircraft_index=encode_aircraft_type(flight.aircraft_type),
            source_index=encode_source(flight.ads_b_source),
            reference_time=float(reference),
        ),
        None,
    )


def _soft_gaussian_label(
    times: np.ndarray, t_td: float, label_sigma_s: float
) -> np.ndarray:
    """A normalised Gaussian-bump target over ``times`` centred on ``t_td``.

    The bump is renormalised to sum to 1 (a proper distribution over timesteps).
    When every weight underflows to ~0 (touchdown far outside the window) the
    mass is placed on the single nearest sample so the target stays valid.
    """
    sigma = max(float(label_sigma_s), 1e-6)
    z = (np.asarray(times, dtype=float) - float(t_td)) / sigma
    weights = np.exp(-0.5 * z * z)
    total = float(weights.sum())
    if total <= 0.0 or not np.isfinite(total):
        out = np.zeros_like(weights)
        out[int(np.argmin(np.abs(np.asarray(times, dtype=float) - float(t_td))))] = 1.0
        return out
    return weights / total


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class SequenceModelEstimator(PhysicsEstimator):
    """TCN sequence-model touchdown estimator (Req 5.3); see module docstring.

    Parameters
    ----------
    label_sigma_s:
        Width of the soft Gaussian label (seconds).
    hidden_dim, aircraft_embed_dim, source_embed_dim, kernel_size, dilations:
        Network shape hyperparameters.
    n_epochs, learning_rate:
        Full-batch training schedule.
    n_ensemble:
        Number of independently-seeded models in the deep ensemble (1 disables).
    seed:
        Master seed propagated to every ensemble member (Req 15.2).
    signals_config:
        :class:`~tdz.config.schema.SignalsConfig` for the derivative channels;
        defaults to the feature module's default.
    deterministic:
        Request PyTorch deterministic algorithms (recorded in diagnostics).
    min_sigma_t_s:
        Floor on the reported ``sigma_t`` (seconds).
    """

    method_name = METHOD_NAME

    def __init__(
        self,
        *,
        label_sigma_s: float = DEFAULT_LABEL_SIGMA_S,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        aircraft_embed_dim: int = DEFAULT_AIRCRAFT_EMBED_DIM,
        source_embed_dim: int = DEFAULT_SOURCE_EMBED_DIM,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dilations: Sequence[int] = DEFAULT_DILATIONS,
        n_epochs: int = DEFAULT_N_EPOCHS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        n_ensemble: int = DEFAULT_N_ENSEMBLE,
        seed: int = DEFAULT_SEED,
        signals_config: Optional[SignalsConfig] = None,
        deterministic: bool = True,
        min_sigma_t_s: float = MIN_SIGMA_T_S,
    ) -> None:
        if label_sigma_s <= 0.0:
            raise ValueError(f"label_sigma_s must be > 0, got {label_sigma_s}")
        if n_ensemble < 1:
            raise ValueError(f"n_ensemble must be >= 1, got {n_ensemble}")
        self.label_sigma_s = float(label_sigma_s)
        self.hidden_dim = int(hidden_dim)
        self.aircraft_embed_dim = int(aircraft_embed_dim)
        self.source_embed_dim = int(source_embed_dim)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(int(d) for d in dilations)
        self.n_epochs = int(n_epochs)
        self.learning_rate = float(learning_rate)
        self.n_ensemble = int(n_ensemble)
        self.seed = int(seed)
        self.signals_config = signals_config
        self.deterministic = bool(deterministic)
        self.min_sigma_t_s = float(min_sigma_t_s)

        self._models: list = []                       # trained torch modules
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._n_train_samples: int = 0
        self._n_train_skipped: int = 0
        #: Per-aircraft-type training-flight counts (Req 6.3 fallback wiring).
        self.training_type_counts: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """``True`` once :meth:`train` has fitted at least one model."""
        return len(self._models) > 0

    # ------------------------------------------------------------------
    # Standardisation
    # ------------------------------------------------------------------

    def _standardise(self, continuous: np.ndarray) -> np.ndarray:
        """Apply the stored per-channel mean/std; NaN -> 0 after centring."""
        mean = self._feature_mean
        std = self._feature_std
        z = (continuous - mean) / std
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _assemble_features(self, seq: SequenceInput) -> np.ndarray:
        """Standardised continuous channels concatenated with the raw flags."""
        cont = self._standardise(seq.continuous)
        return np.concatenate([cont, seq.flags], axis=1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def build_training_sequences(
        self,
        flights: Sequence[FlightRecord],
        truths: Sequence[QARTruthRecord],
    ) -> tuple[list[SequenceInput], list[np.ndarray], list[str]]:
        """Build ``(inputs, soft_labels, skipped_ids)`` from flights + truths.

        Each usable flight yields its :class:`SequenceInput` and a soft Gaussian
        label centred on the matching QAR touchdown time. The QAR truth is used
        ONLY to form the label, never as a model input. Per-type training-flight
        counts (over *usable* flights) are recorded on
        :attr:`training_type_counts` for the rare-type fallback (Req 6.3).
        """
        truth_by_id = {t.flight_id: t for t in truths}
        inputs: list[SequenceInput] = []
        labels: list[np.ndarray] = []
        skipped: list[str] = []
        counts: dict[str, int] = {}

        for flight in flights:
            truth = truth_by_id.get(flight.flight_id)
            if truth is None:
                skipped.append(flight.flight_id)
                continue
            seq, _reason = build_sequence_input(flight, self.signals_config)
            if seq is None:
                skipped.append(flight.flight_id)
                continue
            label = _soft_gaussian_label(
                seq.times, float(truth.touchdown_time_qar), self.label_sigma_s
            )
            inputs.append(seq)
            labels.append(label)
            counts[flight.aircraft_type] = counts.get(flight.aircraft_type, 0) + 1

        self.training_type_counts = counts
        return inputs, labels, skipped

    def _fit_standardiser(self, inputs: Sequence[SequenceInput]) -> None:
        """Compute per-channel mean/std over all training timesteps (NaN-aware)."""
        stacked = np.vstack([seq.continuous for seq in inputs])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(stacked, axis=0)
            std = np.nanstd(stacked, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std[std < 1e-6] = 1.0  # guard constant channels
        self._feature_mean = mean
        self._feature_std = std

    def train(
        self,
        flights: Sequence[FlightRecord],
        truths: Sequence[QARTruthRecord],
    ) -> "SequenceModelEstimator":
        """Fit the sequence model(s) on labeled landings; return ``self``.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        ValueError
            If fewer than :data:`MIN_TRAINING_SAMPLES` usable sequences remain.
        """
        import torch  # local import: optional heavy dependency

        inputs, labels, skipped = self.build_training_sequences(flights, truths)
        if len(inputs) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"need at least {MIN_TRAINING_SAMPLES} usable training sequences, "
                f"got {len(inputs)} (skipped {len(skipped)})"
            )

        self._n_train_samples = len(inputs)
        self._n_train_skipped = len(skipped)
        self._fit_standardiser(inputs)

        # Pre-assemble standardised feature matrices + label tensors once.
        feats = [self._assemble_features(seq) for seq in inputs]
        ac_idx = [seq.aircraft_index for seq in inputs]
        src_idx = [seq.source_index for seq in inputs]

        self._models = [
            self._train_one_model(
                torch, feats, ac_idx, src_idx, labels, member_seed=self.seed + k
            )
            for k in range(self.n_ensemble)
        ]
        return self

    #: ``fit`` is an alias for :meth:`train` (sklearn-style naming).
    fit = train

    def _train_one_model(
        self, torch, feats, ac_idx, src_idx, labels, *, member_seed: int
    ):
        """Train a single ensemble member with full-batch gradient descent."""
        self._seed_torch(torch, member_seed)
        in_channels = _N_CONTINUOUS + _N_FLAGS
        model = _TouchdownTCN(
            torch,
            in_channels=in_channels,
            hidden_dim=self.hidden_dim,
            aircraft_vocab=AIRCRAFT_TYPE_VOCAB_SIZE,
            aircraft_embed_dim=self.aircraft_embed_dim,
            source_vocab=N_SOURCE_CODES,
            source_embed_dim=self.source_embed_dim,
            kernel_size=self.kernel_size,
            dilations=self.dilations,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # Convert once to tensors (via lists -- robust to the torch/numpy bridge).
        feat_tensors = [_to_tensor(torch, f) for f in feats]
        label_tensors = [_to_tensor(torch, lab) for lab in labels]
        ac_tensors = [torch.as_tensor([int(a)], dtype=torch.long) for a in ac_idx]
        src_tensors = [torch.as_tensor([int(s)], dtype=torch.long) for s in src_idx]

        model.train()
        for _epoch in range(self.n_epochs):
            optimizer.zero_grad()
            total = None
            for f_t, lab_t, a_t, s_t in zip(
                feat_tensors, label_tensors, ac_tensors, src_tensors
            ):
                log_p = model(f_t, a_t, s_t)            # (T,) log-probabilities
                # Cross-entropy between the soft label and predicted distribution.
                loss = -(lab_t * log_p).sum()
                total = loss if total is None else total + loss
            total = total / len(feat_tensors)
            total.backward()
            optimizer.step()

        model.eval()
        return model

    def _seed_torch(self, torch, seed: int) -> None:
        """Seed all RNGs for reproducible CPU training (Req 15.1/15.2)."""
        import random

        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if self.deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover - older torch without the flag
                pass

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_distribution(
        self, flight: FlightRecord
    ) -> tuple[Optional[np.ndarray], Optional[SequenceInput], Optional[np.ndarray]]:
        """Return ``(mean_p, sequence_input, per_member_t_td)`` or ``(None, ..)``.

        ``mean_p`` is the ensemble-averaged per-timestep touchdown distribution;
        ``per_member_t_td`` holds each member's expected-value ``t_td`` (for the
        epistemic spread). Returns ``(None, None, None)`` when no sequence can be
        built or the estimator is untrained.
        """
        if not self.is_trained:
            return None, None, None
        seq, _reason = build_sequence_input(flight, self.signals_config)
        if seq is None:
            return None, seq, None

        import torch

        feats = self._assemble_features(seq)
        f_t = _to_tensor(torch, feats)
        a_t = torch.as_tensor([int(seq.aircraft_index)], dtype=torch.long)
        s_t = torch.as_tensor([int(seq.source_index)], dtype=torch.long)

        member_p: list[np.ndarray] = []
        member_t_td: list[float] = []
        with torch.no_grad():
            for model in self._models:
                log_p = model(f_t, a_t, s_t)
                p = np.asarray(torch.exp(log_p).tolist(), dtype=float)
                p = p / p.sum() if p.sum() > 0 else p
                member_p.append(p)
                member_t_td.append(float(np.sum(p * seq.times)))

        mean_p = np.mean(np.vstack(member_p), axis=0)
        mean_p = mean_p / mean_p.sum() if mean_p.sum() > 0 else mean_p
        return mean_p, seq, np.asarray(member_t_td, dtype=float)

    def _raw_estimate(self, flight: FlightRecord) -> TDEstimate:
        """Predict ``t_td`` from the per-timestep distribution; see module docstring."""
        if not self.is_trained:
            return make_estimate(
                t_td=float("nan"),
                sigma_t=float("inf"),
                confidence="failed",
                method_name=self.method_name,
                diagnostics={"detail": "model_not_trained"},
                reason=None,
            )

        # Reuse build_sequence_input for the failure reason if it cannot build.
        seq_probe, reason = build_sequence_input(flight, self.signals_config)
        if seq_probe is None:
            return failed_estimate(
                self.method_name, reason or FailureReason.INSUFFICIENT_SAMPLES
            )

        mean_p, seq, member_t_td = self.predict_distribution(flight)
        assert mean_p is not None and seq is not None

        t_td = float(np.sum(mean_p * seq.times))
        variance = float(np.sum(mean_p * (seq.times - t_td) ** 2))
        width = math.sqrt(max(variance, 0.0))

        # Epistemic term: spread of the ensemble members' expected values.
        if member_t_td is not None and member_t_td.size > 1:
            epistemic = float(np.std(member_t_td))
        else:
            epistemic = 0.0

        sigma_t = max(math.hypot(width, epistemic), self.min_sigma_t_s)
        peak_index = int(np.argmax(mean_p))

        diagnostics = {
            "reference_time": seq.reference_time,
            "expected_t_td": t_td,
            "distribution_width_s": width,
            "epistemic_sigma_s": epistemic,
            "peak_probability": float(mean_p[peak_index]),
            "peak_time": float(seq.times[peak_index]),
            "n_timesteps": int(seq.times.size),
            "n_ensemble": int(self.n_ensemble),
            "label_sigma_s": self.label_sigma_s,
            "deterministic_mode": self.deterministic,
            "n_train_samples": self._n_train_samples,
        }
        return make_estimate(
            t_td=t_td,
            sigma_t=sigma_t,
            confidence=CONFIDENCE_NORMAL,
            method_name=self.method_name,
            diagnostics=diagnostics,
            reason=None,
        )


# ---------------------------------------------------------------------------
# torch helpers / model (built lazily so the module imports without torch)
# ---------------------------------------------------------------------------


def _to_tensor(torch, array: np.ndarray):
    """Build a float32 tensor from a numpy array.

    Prefers the zero-copy ``torch.from_numpy`` fast path, falling back to a
    list round-trip when the torch<->numpy bridge is unavailable (e.g. a torch
    build compiled against a different NumPy ABI). The fallback is correct,
    just slower, and keeps the estimator runnable across torch/NumPy combos.

    LOCAL DEV WORKAROUND -- NOT production-required.
    --------------------------------------------------
    The fallback branch exists only to support local development on Intel
    macOS (x86_64), where the newest installable PyTorch is 2.2.2 (Apple
    dropped Intel-Mac wheels after it). That build was compiled against the
    NumPy 1.x C-ABI, so its ``from_numpy`` bridge fails to initialize under
    NumPy 2.x and raises -- hence the list round-trip.

    Production runs on Linux (Azure Databricks), where torch 2.3+ supports
    NumPy 2.x and the zero-copy fast path is always taken; the fallback is
    never exercised there. Do NOT treat this shim as a pattern to copy into
    production code paths, and do NOT add a project-wide ``numpy<2`` pin on
    its account -- the constraint is specific to the Intel-Mac dev box. See
    the "Local development on Intel macOS" note in the README.
    """
    a = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    try:
        return torch.from_numpy(a).float()
    except (RuntimeError, TypeError):
        return torch.as_tensor(a.tolist(), dtype=torch.float32)


def _TouchdownTCN(torch, **kwargs):
    """Construct the TCN module (defined lazily to avoid importing ``nn`` early)."""
    import torch.nn as nn
    import torch.nn.functional as F

    class _Module(nn.Module):
        """Dilated 1-D TCN over the landing window -> per-timestep logits.

        Each timestep's input is the standardised channel vector concatenated
        with the (broadcast) aircraft-type and source embeddings. Same-padded
        dilated convolutions give every timestep a window-spanning receptive
        field (touchdown sits mid-sequence), and a final 1x1 conv produces one
        logit per timestep; a softmax over time yields ``P(touchdown)``.
        """

        def __init__(
            self,
            *,
            in_channels: int,
            hidden_dim: int,
            aircraft_vocab: int,
            aircraft_embed_dim: int,
            source_vocab: int,
            source_embed_dim: int,
            kernel_size: int,
            dilations,
        ) -> None:
            super().__init__()
            self.aircraft_embed = nn.Embedding(aircraft_vocab, aircraft_embed_dim)
            self.source_embed = nn.Embedding(source_vocab, source_embed_dim)
            proj_in = in_channels + aircraft_embed_dim + source_embed_dim
            self.input_proj = nn.Linear(proj_in, hidden_dim)
            self.convs = nn.ModuleList(
                [
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=((kernel_size - 1) // 2) * d,
                        dilation=d,
                    )
                    for d in dilations
                ]
            )
            self.output_conv = nn.Conv1d(hidden_dim, 1, kernel_size=1)

        def forward(self, features, aircraft_index, source_index):
            # features: (T, C); embeddings: (1, E) -> broadcast across T.
            t_steps = features.shape[0]
            ac = self.aircraft_embed(aircraft_index).expand(t_steps, -1)
            src = self.source_embed(source_index).expand(t_steps, -1)
            x = torch.cat([features, ac, src], dim=1)        # (T, proj_in)
            h = self.input_proj(x)                           # (T, hidden)
            h = h.transpose(0, 1).unsqueeze(0)               # (1, hidden, T)
            for conv in self.convs:
                h = h + F.relu(conv(h))                      # residual TCN block
            logits = self.output_conv(h).squeeze(0).squeeze(0)  # (T,)
            return F.log_softmax(logits, dim=0)

    return _Module(**kwargs)
