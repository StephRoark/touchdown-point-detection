"""IMM filter + RTS smoother physics estimator (Task 12.3).

Estimates touchdown as the **mode-probability crossover** of a two-mode
Interacting-Multiple-Model (IMM) treatment of the landing kinematics, sharpened
by a backward smoother (design "IMM Filter + RTS Smoother"; Req 5.1, 6.1, 18).

The two modes (design)
----------------------
* **Mode 1 -- descending / approach dynamics:** gentle longitudinal
  deceleration and a clearly negative vertical rate (the aircraft is on the
  glideslope, still coming down).
* **Mode 2 -- ground roll:** higher longitudinal deceleration (wheel braking /
  spoilers / reversers) and ~zero vertical rate (the aircraft is on the runway).

As the aircraft touches down the most-likely mode swings from 1 to 2. The time
at which ``P(mode 2)`` overtakes ``P(mode 1)`` -- i.e. crosses ``0.5`` -- is the
touchdown estimate, taken to **sub-sample resolution** by linearly interpolating
the crossing between the two bracketing update epochs (design: "Mode-probability
crossover gives ``t_td`` to sub-sample resolution").

State vector and dynamics (continuous-time, numpy-only)
-------------------------------------------------------
A single shared kinematic Kalman filter tracks

    x = [v, a, h, w]

where ``v`` is along-track ground speed (m/s), ``a`` is along-track acceleration
(m/s^2, negative under braking), ``h`` is height above the runway (m, HAE), and
``w`` is vertical rate (m/s). The continuous-time predict over an inter-event gap
``dt`` is the constant-acceleration / constant-rate transition

    F(dt) = [[1, dt, 0,  0],
             [0, 1, 0,  0],
             [0, 0, 1, dt],
             [0, 0, 0,  1]]

with a white-noise-jerk process model on the ``(v, a)`` pair (PSD
:data:`Q_LONG_JERK_PSD`) and a white-noise-vertical-acceleration model on the
``(h, w)`` pair (PSD :data:`Q_VERT_ACCEL_PSD`); each uses the standard
continuous-white-noise-acceleration discretization ``q * [[dt^3/3, dt^2/2],
[dt^2/2, dt]]``. Measurements are consumed at their **native** timestamps -- the
groundspeed channel (velocity timebase) updates ``v`` and the geometric-altitude
channel (position timebase) updates ``h`` -- merged into one time-ordered event
stream so the async position/velocity samples are never co-timed or merged
(design: "Consumes async position/velocity natively via continuous-time state
updates"; Req 10.1). No filterpy/scipy: the 4-state filter is a few small numpy
matrix ops.

Two-mode likelihood filter (documented IMM reduction)
-----------------------------------------------------
A full IMM mixes *mode-conditioned* state filters. Here the continuous kinematic
state is estimated by the single shared filter above and the two **modes** are a
two-state hidden-Markov chain over the mode label with Gaussian emission
likelihoods on the filtered deceleration and vertical rate. This is a principled
reduction of the full IMM to a **two-mode likelihood filter** (the mode mixing
acts on the discrete label rather than on parallel state filters); it is far
simpler and numerically robust at the 4-5 s ADS-B cadence while preserving the
quantity that matters -- the mode-probability crossover -- to sub-sample
resolution. The emission regimes are **adaptive** (data-driven low/high
deceleration and descent-rate quantiles, with documented separation floors) so
the crossover lands at the flight's own approach->rollout transition rather than
relying on absolute calibration. When the source lacks geometric altitude the
vertical channel is dropped and discrimination uses the longitudinal channel
alone (the estimator still runs -- it is not a vertical-only estimator).

Forward filtering uses a sticky Markov transition matrix (approach is sticky,
ground roll is near-absorbing -- a landing transitions forward, essentially
once). The **backward smoother** is the analog of the RTS pass for a discrete
mode chain: the standard Baum forward-backward recursion, which sharpens the
mode transition (design: "The backward RTS smoother sharpens the transition").

How ``t_td`` and ``sigma_t`` are computed
-----------------------------------------
``t_td`` is the smoothed ``P(mode 2)=0.5`` crossing, linearly interpolated
between the bracketing epochs (sub-sample). ``sigma_t`` is derived from the
**crossover sharpness** -- how fast ``P(mode 2)`` swings through ``0.5`` (per
second): a sharp swing localizes the crossing tightly, a shallow one widens it.
The formula mirrors the other physics estimators:

    sigma_t = hypot(PROB_SIGMA / max(|sharpness|, MIN_SHARPNESS_PER_S),
                    CADENCE_SIGMA_FRACTION * median_dt)

floored at :data:`MIN_SIGMA_T_S`. ``PROB_SIGMA`` is the residual 1-sigma noise on
the mode probability; dividing by the sharpness (probability/second) maps it to a
time. The smoother state covariance at ``t_td`` is also surfaced in diagnostics.

Confidence / failure
--------------------
* No groundspeed at all -> failed with
  :attr:`~tdz.models.FailureReason.NO_GROUNDSPEED`.
* Fewer than :data:`MIN_USABLE_SAMPLES` usable groundspeed samples -> failed with
  :attr:`~tdz.models.FailureReason.INSUFFICIENT_SAMPLES`.
* The mode never crosses into ground roll (no ``0.5`` crossing) -> failed with
  :attr:`~tdz.models.FailureReason.NO_GROUND_ROLL_CONFIRMATION` (no ground-roll
  regime was confirmed -- e.g. a go-around-like profile).
* Otherwise ``"normal"``.

The Requirement-18 on-ground upper bound is applied by
:class:`~tdz.estimators.physics.base.PhysicsEstimator`; this estimator does
**not** re-clamp.

Diagnostics: mode probabilities over time (filtered and smoothed) with their
epoch times, the crossover time and sharpness, the smoother innovation and state
covariance at ``t_td``, and the filtered state at ``t_td``.

Units: SI throughout (m, m/s, m/s^2, s). Groundspeed knots->m/s via
:data:`~tdz.timebase.interpolation.KNOTS_TO_MPS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

import numpy as np

from tdz.estimators.physics.base import (
    CONFIDENCE_NORMAL,
    PhysicsEstimator,
    failed_estimate,
    make_estimate,
)
from tdz.geo.datum import resolve_threshold_elevation_hae
from tdz.geo.errors import DatumUnresolvedError
from tdz.models import FailureReason, FlightRecord, TDEstimate
from tdz.timebase.interpolation import KNOTS_TO_MPS

__all__ = [
    "ImmRtsEstimator",
    "METHOD_NAME",
    "MIN_USABLE_SAMPLES",
    "MIN_VERTICAL_UPDATES",
    "Q_LONG_JERK_PSD",
    "Q_VERT_ACCEL_PSD",
    "SIGMA_V_MPS",
    "SIGMA_H_M",
    "MARKOV_STAY_APPROACH",
    "MARKOV_STAY_GROUND",
    "INIT_GROUND_PROB",
    "PROB_SIGMA",
    "MIN_SHARPNESS_PER_S",
    "CADENCE_SIGMA_FRACTION",
    "MIN_SIGMA_T_S",
]

#: Estimator identifier (matches the ``imm_rts`` id in ``ALLOWED_ESTIMATORS``).
METHOD_NAME: Final[str] = "imm_rts"

#: Minimum number of usable (finite) groundspeed samples required to run the
#: filter and locate a crossover. Below this the profile cannot support a
#: two-regime transition and the estimator fails INSUFFICIENT_SAMPLES.
MIN_USABLE_SAMPLES: Final[int] = 4

#: Minimum number of geometric-altitude updates before the vertical channel is
#: trusted as a mode discriminator. Below this (e.g. an FR24-like velocity-only
#: source) the estimator uses the longitudinal channel alone.
MIN_VERTICAL_UPDATES: Final[int] = 3

#: Process-noise PSD for the white-noise-jerk model on the (v, a) pair
#: ((m/s^2)^2 / s). Sized so the tracked deceleration can change by ~0.5-1 m/s^2
#: across a 4-5 s gap, i.e. follow the approach->rollout knee within a sample.
Q_LONG_JERK_PSD: Final[float] = 0.2

#: Process-noise PSD for the white-noise-vertical-acceleration model on the
#: (h, w) pair ((m/s^2)^2 / s).
Q_VERT_ACCEL_PSD: Final[float] = 0.5

#: Groundspeed measurement 1-sigma (m/s) used by the shared filter's v-update.
SIGMA_V_MPS: Final[float] = 2.0

#: Geometric-altitude measurement 1-sigma (m) used by the shared filter's
#: h-update.
SIGMA_H_M: Final[float] = 8.0

#: Sticky Markov self-transition for the approach mode (mode 1 stays mode 1).
MARKOV_STAY_APPROACH: Final[float] = 0.90

#: Sticky Markov self-transition for the ground-roll mode (near-absorbing: a
#: landing transitions forward essentially once and does not revert).
MARKOV_STAY_GROUND: Final[float] = 0.99

#: Initial probability of being already in ground roll at the first epoch (small
#: -- a landing starts on approach).
INIT_GROUND_PROB: Final[float] = 0.02

#: Residual 1-sigma noise on the mode probability, used in the sigma_t mapping
#: ``PROB_SIGMA / sharpness`` (probability / (probability/s) = s).
PROB_SIGMA: Final[float] = 0.10

#: Floor on the crossover sharpness (probability/second) so an almost-flat
#: crossing does not produce an absurd sigma_t.
MIN_SHARPNESS_PER_S: Final[float] = 1e-3

#: Fraction of the median event spacing used as the cadence floor on sigma_t.
CADENCE_SIGMA_FRACTION: Final[float] = 0.5

#: Absolute floor on the reported sigma_t (seconds).
MIN_SIGMA_T_S: Final[float] = 0.25

# --- adaptive emission-regime tunables (documented in the module docstring) ---

#: Low/high deceleration quantiles (percent) used as the approach / ground-roll
#: longitudinal mode targets.
_LONG_LOW_PCT: Final[float] = 20.0
_LONG_HIGH_PCT: Final[float] = 90.0

#: Minimum separation (m/s^2) enforced between the two longitudinal targets so a
#: nearly constant-deceleration profile still yields a meaningful crossover.
_MIN_DECEL_SEPARATION_MPS2: Final[float] = 0.5

#: Emission std as a fraction of the target separation, with an absolute floor.
_LONG_SD_FRACTION: Final[float] = 0.6
_MIN_LONG_SD_MPS2: Final[float] = 0.25

#: Vertical-channel descent quantile (percent; most-negative tail) and floors.
_VERT_LOW_PCT: Final[float] = 20.0
_MIN_DESCENT_SEPARATION_MPS: Final[float] = 0.8
_VERT_SD_FRACTION: Final[float] = 0.6
_MIN_VERT_SD_MPS: Final[float] = 0.4


@dataclass(frozen=True)
class _FilterTrace:
    """Per-epoch record of the shared kinematic filter (all SI)."""

    times: np.ndarray            # (E,) event epoch seconds (sorted)
    states: np.ndarray           # (E, 4) filtered [v, a, h, w]
    cov_diag: np.ndarray         # (E, 4) filtered state-covariance diagonal
    innovations: np.ndarray      # (E,) groundspeed innovation (NaN on h-events)
    n_vertical_updates: int      # number of height updates applied


class ImmRtsEstimator(PhysicsEstimator):
    """Estimate ``t_td`` from the IMM mode-probability crossover (Req 5.1, 6.1).

    Parameters
    ----------
    geodesy_config:
        Optional geodesy config passed to the datum resolver (only used when the
        source carries geometric altitude).
    sigma_v_mps, sigma_h_m:
        Measurement 1-sigma for the groundspeed / geometric-altitude updates.
    q_long_jerk_psd, q_vert_accel_psd:
        Process-noise PSDs for the longitudinal / vertical blocks.
    cadence_sigma_fraction, min_sigma_t_s:
        Overridable ``sigma_t`` tunables (see module docstring / constants).
    """

    method_name = METHOD_NAME

    def __init__(
        self,
        *,
        geodesy_config: object = None,
        sigma_v_mps: float = SIGMA_V_MPS,
        sigma_h_m: float = SIGMA_H_M,
        q_long_jerk_psd: float = Q_LONG_JERK_PSD,
        q_vert_accel_psd: float = Q_VERT_ACCEL_PSD,
        cadence_sigma_fraction: float = CADENCE_SIGMA_FRACTION,
        min_sigma_t_s: float = MIN_SIGMA_T_S,
    ) -> None:
        self.geodesy_config = geodesy_config
        self.sigma_v_mps = float(sigma_v_mps)
        self.sigma_h_m = float(sigma_h_m)
        self.q_long_jerk_psd = float(q_long_jerk_psd)
        self.q_vert_accel_psd = float(q_vert_accel_psd)
        self.cadence_sigma_fraction = float(cadence_sigma_fraction)
        self.min_sigma_t_s = float(min_sigma_t_s)

    # -- public estimator contract ------------------------------------------

    def _raw_estimate(self, flight: FlightRecord) -> TDEstimate:
        events = self._build_events(flight)
        if events is None:
            return failed_estimate(self.method_name, FailureReason.NO_GROUNDSPEED)

        n_speed = int(np.count_nonzero(events[:, 1] == 0.0))
        if n_speed < MIN_USABLE_SAMPLES:
            return failed_estimate(self.method_name, FailureReason.INSUFFICIENT_SAMPLES)

        trace = self._run_filter(events)
        emissions = self._mode_emissions(trace)
        p_filt = self._forward(emissions)
        p_smooth = self._smooth(emissions, p_filt)

        crossover = self._find_crossover(trace.times, p_smooth[:, 1])
        if crossover is None:
            # Ground roll never overtakes approach: no touchdown transition
            # detected (e.g. a go-around-like profile).
            return failed_estimate(
                self.method_name,
                FailureReason.NO_GROUND_ROLL_CONFIRMATION,
                diagnostics={
                    "mode_probability_times": [float(t) for t in trace.times],
                    "mode_probabilities_smoothed": p_smooth.tolist(),
                    "reason_detail": "P(mode 2) never crossed 0.5",
                },
            )

        k, t_td, sharpness = crossover
        sigma_t = self._sigma_t(sharpness, trace.times)

        state_td = self._interp_state(trace, k, t_td)
        cov_td = trace.cov_diag[k]
        innovation_td = self._innovation_near(trace, t_td)

        diagnostics = {
            "crossover_time": t_td,
            "crossover_sharpness_per_s": sharpness,
            "mode_probability_times": [float(t) for t in trace.times],
            "mode_probabilities_filtered": p_filt.tolist(),
            "mode_probabilities_smoothed": p_smooth.tolist(),
            "p_ground_at_crossover": (
                float(p_smooth[k - 1, 1]),
                float(p_smooth[k, 1]),
            ),
            "state_at_td": {
                "along_track_speed_mps": state_td[0],
                "along_track_accel_mps2": state_td[1],
                "height_above_runway_m": state_td[2],
                "vertical_rate_mps": state_td[3],
            },
            "smoother_covariance_diag_at_td": {
                "var_v": float(cov_td[0]),
                "var_a": float(cov_td[1]),
                "var_h": float(cov_td[2]),
                "var_w": float(cov_td[3]),
            },
            "smoother_innovation_at_td_mps": innovation_td,
            "n_vertical_updates": trace.n_vertical_updates,
            "vertical_channel_used": trace.n_vertical_updates >= MIN_VERTICAL_UPDATES,
            "n_events": int(trace.times.size),
        }
        return make_estimate(
            t_td=t_td,
            sigma_t=sigma_t,
            confidence=CONFIDENCE_NORMAL,
            method_name=self.method_name,
            diagnostics=diagnostics,
        )

    # -- event construction --------------------------------------------------

    def _build_events(self, flight: FlightRecord) -> Optional[np.ndarray]:
        """Merge groundspeed and (optional) height samples into one event stream.

        Returns an ``(E, 3)`` array of rows ``[time, kind, value]`` sorted by
        time, where ``kind`` is ``0.0`` for a groundspeed event (value in m/s)
        and ``1.0`` for a height event (value in metres above the runway), or
        ``None`` when there is no usable groundspeed at all.
        """
        vt = np.asarray(flight.velocity_times, dtype=float)
        gs_kt = np.asarray(flight.groundspeeds, dtype=float)
        rows = []
        if gs_kt.size and not np.all(np.isnan(gs_kt)):
            for t, g in zip(vt, gs_kt):
                if np.isfinite(t) and np.isfinite(g):
                    rows.append((float(t), 0.0, float(g) * KNOTS_TO_MPS))
        if not rows:
            return None

        heights = self._height_samples(flight)
        if heights is not None:
            pt, h = heights
            for t, hv in zip(pt, h):
                if np.isfinite(t) and np.isfinite(hv):
                    rows.append((float(t), 1.0, float(hv)))

        arr = np.array(rows, dtype=float)
        order = np.argsort(arr[:, 0], kind="stable")
        return arr[order]

    def _height_samples(
        self, flight: FlightRecord
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Height-above-runway samples (HAE) on the position timebase, or None.

        Returns ``None`` when the source carries no geometric altitude or the
        runway datum cannot be resolved -- the estimator then runs velocity-only.
        """
        geo_alt = np.asarray(flight.geometric_altitudes, dtype=float)
        if geo_alt.size == 0 or np.all(np.isnan(geo_alt)):
            return None
        try:
            threshold_hae = resolve_threshold_elevation_hae(
                flight.runway, self.geodesy_config
            )
        except DatumUnresolvedError:
            return None
        pt = np.asarray(flight.position_times, dtype=float)
        return pt, geo_alt - threshold_hae

    # -- shared continuous-time kinematic filter -----------------------------

    def _run_filter(self, events: np.ndarray) -> _FilterTrace:
        """Run the shared 4-state continuous-time Kalman filter over the events.

        Events sharing a timestamp (co-timed sources) are grouped into a single
        epoch -- the predict runs once over the gap and every measurement at that
        instant is applied -- so the recorded epoch times are strictly
        increasing (no zero-span brackets in the crossover search).
        """
        # Initial state from the first groundspeed / height events.
        speed_vals = events[events[:, 1] == 0.0, 2]
        height_rows = events[events[:, 1] == 1.0]
        v0 = float(speed_vals[0])
        h0 = float(height_rows[0, 2]) if height_rows.size else 0.0

        x = np.array([v0, 0.0, h0, 0.0], dtype=float)
        P = np.diag(
            [self.sigma_v_mps**2, 4.0, self.sigma_h_m**2, 4.0]
        ).astype(float)

        unique_times = np.unique(events[:, 0])
        n = unique_times.size
        states = np.empty((n, 4), dtype=float)
        cov_diag = np.empty((n, 4), dtype=float)
        innovations = np.full(n, np.nan, dtype=float)
        n_vertical = 0

        prev_t = float(unique_times[0])
        for ei, tu in enumerate(unique_times):
            dt = float(tu) - prev_t
            if dt > 0.0:
                x, P = self._predict(x, P, dt)
            prev_t = float(tu)

            rows = events[events[:, 0] == tu]
            for row in rows:
                kind, value = row[1], float(row[2])
                if kind == 0.0:  # groundspeed update of v
                    innovations[ei] = value - x[0]
                    x, P = self._update(x, P, h_index=0, z=value, r=self.sigma_v_mps**2)
                else:  # height update of h
                    x, P = self._update(x, P, h_index=2, z=value, r=self.sigma_h_m**2)
                    n_vertical += 1

            states[ei] = x
            cov_diag[ei] = np.diag(P)

        return _FilterTrace(
            times=unique_times.copy(),
            states=states,
            cov_diag=cov_diag,
            innovations=innovations,
            n_vertical_updates=n_vertical,
        )

    def _predict(
        self, x: np.ndarray, P: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Continuous-time constant-accel / constant-rate predict over ``dt``."""
        F = np.eye(4)
        F[0, 1] = dt
        F[2, 3] = dt
        Q = np.zeros((4, 4))
        Q[0:2, 0:2] = self._cwna_block(self.q_long_jerk_psd, dt)
        Q[2:4, 2:4] = self._cwna_block(self.q_vert_accel_psd, dt)
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred

    @staticmethod
    def _cwna_block(psd: float, dt: float) -> np.ndarray:
        """Continuous-white-noise-acceleration process-noise block for a pair."""
        dt2 = dt * dt
        dt3 = dt2 * dt
        return psd * np.array([[dt3 / 3.0, dt2 / 2.0], [dt2 / 2.0, dt]])

    @staticmethod
    def _update(
        x: np.ndarray, P: np.ndarray, *, h_index: int, z: float, r: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scalar Kalman update measuring the state component at ``h_index``."""
        H = np.zeros((1, 4))
        H[0, h_index] = 1.0
        y = z - x[h_index]
        S = float(P[h_index, h_index] + r)
        K = (P @ H.T).reshape(4) / S
        x_new = x + K * y
        P_new = (np.eye(4) - np.outer(K, H[0])) @ P
        # Symmetrize for numerical hygiene.
        P_new = 0.5 * (P_new + P_new.T)
        return x_new, P_new

    # -- two-mode emission likelihoods ---------------------------------------

    def _mode_emissions(self, trace: _FilterTrace) -> np.ndarray:
        """Per-epoch ``(E, 2)`` mode likelihoods (columns: approach, ground roll).

        Adaptive longitudinal targets (low/high deceleration quantiles) and, when
        the vertical channel is trusted, adaptive descent / level targets. The
        likelihoods are normalized per epoch (subtract the max log-likelihood)
        for numerical stability before exponentiation.
        """
        decel = -trace.states[:, 1]          # positive under braking
        vrate = trace.states[:, 3]           # vertical rate (negative descending)

        d_lo, d_hi, sd_d = self._long_regime(decel)
        ll1 = -0.5 * ((decel - d_lo) / sd_d) ** 2
        ll2 = -0.5 * ((decel - d_hi) / sd_d) ** 2

        if trace.n_vertical_updates >= MIN_VERTICAL_UPDATES:
            w_descent, sd_w = self._vert_regime(vrate)
            ll1 = ll1 - 0.5 * ((vrate - w_descent) / sd_w) ** 2
            ll2 = ll2 - 0.5 * ((vrate - 0.0) / sd_w) ** 2

        ll = np.stack([ll1, ll2], axis=1)
        ll = ll - ll.max(axis=1, keepdims=True)
        emissions = np.exp(ll)
        return emissions

    @staticmethod
    def _long_regime(decel: np.ndarray) -> tuple[float, float, float]:
        """Adaptive approach/ground-roll deceleration targets and emission std."""
        d_lo = float(np.percentile(decel, _LONG_LOW_PCT))
        d_hi = float(np.percentile(decel, _LONG_HIGH_PCT))
        if d_hi - d_lo < _MIN_DECEL_SEPARATION_MPS2:
            center = 0.5 * (d_lo + d_hi)
            d_lo = center - 0.5 * _MIN_DECEL_SEPARATION_MPS2
            d_hi = center + 0.5 * _MIN_DECEL_SEPARATION_MPS2
        sd_d = max(_LONG_SD_FRACTION * (d_hi - d_lo), _MIN_LONG_SD_MPS2)
        return d_lo, d_hi, sd_d

    @staticmethod
    def _vert_regime(vrate: np.ndarray) -> tuple[float, float]:
        """Adaptive descent target (most-negative tail) and emission std."""
        w_descent = float(np.percentile(vrate, _VERT_LOW_PCT))
        if -w_descent < _MIN_DESCENT_SEPARATION_MPS:
            w_descent = -_MIN_DESCENT_SEPARATION_MPS
        sd_w = max(_VERT_SD_FRACTION * abs(w_descent), _MIN_VERT_SD_MPS)
        return w_descent, sd_w

    # -- forward filter / backward smoother over the mode label --------------

    def _transition_matrix(self) -> np.ndarray:
        """Sticky 2-state Markov transition matrix ``T[i, j] = P(j | i)``."""
        a = MARKOV_STAY_APPROACH
        g = MARKOV_STAY_GROUND
        return np.array([[a, 1.0 - a], [1.0 - g, g]])

    def _forward(self, emissions: np.ndarray) -> np.ndarray:
        """Forward HMM filter -> ``(E, 2)`` filtered mode posteriors."""
        T = self._transition_matrix()
        n = emissions.shape[0]
        post = np.empty((n, 2), dtype=float)

        prior = np.array([1.0 - INIT_GROUND_PROB, INIT_GROUND_PROB])
        cur = prior * emissions[0]
        post[0] = cur / cur.sum()
        for k in range(1, n):
            prior = T.T @ post[k - 1]
            cur = prior * emissions[k]
            total = cur.sum()
            post[k] = cur / total if total > 0.0 else np.array([0.5, 0.5])
        return post

    def _smooth(self, emissions: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Backward (Baum) smoother -> ``(E, 2)`` smoothed mode posteriors.

        The discrete-mode analog of the RTS smoother: it sharpens the transition
        by folding in the future evidence.
        """
        T = self._transition_matrix()
        n = emissions.shape[0]
        beta = np.ones((n, 2), dtype=float)
        for k in range(n - 2, -1, -1):
            b = T @ (emissions[k + 1] * beta[k + 1])
            total = b.sum()
            beta[k] = b / total if total > 0.0 else np.array([0.5, 0.5])

        gamma = post * beta
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return gamma / row_sums

    # -- crossover + uncertainty ---------------------------------------------

    @staticmethod
    def _find_crossover(
        times: np.ndarray, p_ground: np.ndarray
    ) -> Optional[tuple[int, float, float]]:
        """First sub-sample ``P(mode 2)=0.5`` up-crossing.

        Returns ``(k, t_td, sharpness)`` where ``k`` is the upper bracketing
        index, ``t_td`` the linearly interpolated crossing time, and
        ``sharpness`` the probability slope (per second) through ``0.5``. Returns
        ``None`` when no up-crossing exists.
        """
        for k in range(1, p_ground.size):
            p0, p1 = float(p_ground[k - 1]), float(p_ground[k])
            if p0 < 0.5 <= p1:
                t0, t1 = float(times[k - 1]), float(times[k])
                dp = p1 - p0
                frac = (0.5 - p0) / dp if dp != 0.0 else 0.0
                t_td = t0 + frac * (t1 - t0)
                span = t1 - t0
                sharpness = dp / span if span > 0.0 else 0.0
                return k, t_td, sharpness
        return None

    def _sigma_t(self, sharpness: float, times: np.ndarray) -> float:
        """Map crossover sharpness (+ cadence floor) to a time uncertainty."""
        slope = max(abs(sharpness), MIN_SHARPNESS_PER_S)
        fit_term = PROB_SIGMA / slope

        finite = times[np.isfinite(times)]
        if finite.size >= 2:
            median_dt = float(np.median(np.diff(np.sort(finite))))
        else:
            median_dt = 0.0
        cadence_floor = self.cadence_sigma_fraction * median_dt

        sigma = float(np.hypot(fit_term, cadence_floor))
        return max(sigma, self.min_sigma_t_s)

    @staticmethod
    def _interp_state(trace: _FilterTrace, k: int, t_td: float) -> np.ndarray:
        """Linearly interpolate the filtered state at ``t_td`` (epochs k-1, k)."""
        t0, t1 = float(trace.times[k - 1]), float(trace.times[k])
        span = t1 - t0
        frac = (t_td - t0) / span if span > 0.0 else 0.0
        s0, s1 = trace.states[k - 1], trace.states[k]
        return s0 + frac * (s1 - s0)

    @staticmethod
    def _innovation_near(trace: _FilterTrace, t_td: float) -> float:
        """Groundspeed innovation of the speed event nearest ``t_td``."""
        finite = np.isfinite(trace.innovations)
        if not np.any(finite):
            return float("nan")
        idx = np.where(finite)[0]
        nearest = idx[int(np.argmin(np.abs(trace.times[idx] - t_td)))]
        return float(trace.innovations[nearest])
