# Touchdown Point Inference from ADS-B — Design

**Companion to:** REQUIREMENTS.md
**Status:** Draft v1.0
**Scope:** Offline batch inference and validation over 40,500 QAR-matched flights.

---

## 1. Design Philosophy

Three commitments shape the design:

1. **Estimate a time, then map to a place.** Touchdown almost always falls *between* ADS-B samples at 4–5 s cadence. Every estimator therefore targets a sub-sample **touchdown time** `t_td`; the horizontal position is obtained afterward by interpolating the trajectory at `t_td` and projecting onto the runway. This cleanly separates *detection* from *geolocation* and lets all methods share the same back-end mapping and validation.

2. **Physics for trust, learning for accuracy.** With 40,500 labels a deep sequence model is expected to be the most accurate estimator, but a safety case needs explainability. The design always pairs the learned estimator with an interpretable physical anchor, and feeds physics-derived signals into the learned model so it refines physics rather than rediscovering it.

3. **Multiple estimators, calibrated fusion.** Methods fail on different landings (null vertical rate, geometric-altitude bias, long/balked flares, source quirks). An ensemble calibrated on QAR reduces variance and, crucially, the tails that drive overrun risk.

---

## 2. System Architecture

```
                ┌────────────────────────────────────────────────────┐
   ADS-B ──▶    │ 1. Ingest & QA  →  2. Time-base & Resample          │
   Runway ─▶    │ 3. Feature / Signal construction                    │
   AC type ▶    └───────────────┬────────────────────────────────────┘
                                │  (per-flight aligned record)
        ┌───────────────────────┼───────────────────────────┐
        ▼                       ▼                            ▼
  4a. Physics /           4b. Change-point            4c. Learned estimators
      state-estimation        detectors                   (LightGBM, TCN/BiLSTM)
      (decel-knee, IMM)       (CUSUM/PELT, jerk)
        └───────────────────────┼───────────────────────────┘
                                ▼
                     5. Fusion / ensemble  →  t_td + uncertainty
                                ▼
                     6. Time → position mapping (project to runway)
                                ▼
                     7. Validation harness (grouped, stratified, vs QAR)
```

Modules are independent and swappable; each estimator emits a common `(t_td, sigma_t, diagnostics)` contract so the fusion and mapping layers are method-agnostic.

---

## 3. Data Pipeline (Modules 1–3)

### 3.1 Ingest & QA
- Parse per-flight ADS-B and join runway + aircraft-type metadata.
- Deduplicate repeated samples; drop or flag impossible jumps (position/speed outliers) using kinematic gates (max plausible acceleration/turn).
- Record per-flight QA metrics (sample count near touchdown, max gap, fraction of null vertical rate). These feed FR-10 (low-confidence flagging).

### 3.2 Time-base and resampling (critical — see Errors §9.1)
- **Preserve native timestamps.** Position and velocity have different emission times (source 1). Do **not** join on nearest timestamp as if co-timed.
- Two supported strategies, configurable:
  - **(a) Common-grid resampling with kinematic interpolation.** Resample onto a fixed fine grid. Propagate position using the velocity message (dead-reckoning) rather than naive linear lat/long interpolation; interpolate altitude with a shape-aware (monotone / spline) interpolator.
  - **(b) Continuous-time consumption.** Keep measurements at native timestamps and let continuous-time estimators (Kalman/IMM, GP) consume each at its own time. Preferred for the state-estimation branch.
- Emit an explicit **time-delta channel** so any learned model is aware of irregular spacing.

### 3.3 Signal / feature construction
- **Smoothed derivatives via Savitzky–Golay or a Gaussian-process fit**, not raw finite differencing — double-differencing 4–5 s data for jerk amplifies noise catastrophically (Errors §9.3). The GP option additionally yields derivative uncertainty.
- Channels produced per flight:
  - Groundspeed and its 1st (deceleration) and 2nd (jerk) derivatives, smoothed.
  - Geometric altitude above threshold elevation; baro altitude (for cross-check only).
  - Vertical rate: use reported baro vertical rate where present, else derive from smoothed geometric altitude.
  - Track vs runway heading (alignment error).
  - On-ground flag and time-since-flag.
  - Distance-to-threshold along centerline (from projected position).
  - Inter-sample time deltas.
  - Static context: aircraft type, ADS-B source, runway length, approach speed band.

---

## 4. Estimators

All estimators output `(t_td, sigma_t, diagnostics)`.

### 4.1 Physics / state-estimation branch (interpretable anchors)

**4.1.1 Deceleration-knee (primary physical baseline).**
Fit a piecewise model to groundspeed vs time: a gentle approach-deceleration segment, then a sharp ground-roll deceleration segment (optionally a 3-segment approach/transition/rollout). The breakpoint is `t_td`. Robust, lives entirely in the velocity stream (immune to altitude-source and async issues). Aircraft-type priors constrain plausible approach speed and deceleration.

**4.1.2 Vertical flare-crossing.**
Model geometric altitude as a ~3° glideslope curving into a flare (linear + quadratic/exponential), solve for the time it crosses `runway_elevation + lever_arm_height`. Apply the **antenna-to-gear height correction** per aircraft type. Do **not** fit a single straight line through the flare (biases long — Errors §9.4).

**4.1.3 IMM (Interacting Multiple Model) filter + RTS smoother.**
Two modes — descending flight and ground roll — over a continuous-time state. The **mode-probability crossover gives `t_td` to sub-sample resolution**; the backward RTS smoother sharpens it. Consumes async position/velocity natively and handles null vertical rate through the measurement update. This is the strongest pure state-estimation estimator and the recommended physical anchor.

### 4.2 Change-point branch

**4.2.1 Jerk-onset.** Peak of the smoothed jerk of groundspeed marks brake/spoiler onset. Use jerk **onset**, not deceleration magnitude — peak braking lags touchdown (Errors §9.5).

**4.2.2 CUSUM / PELT / GLRT.** Detect the regime change in deceleration. PELT is exact and fast for offline whole-landing segmentation; CUSUM gives an online-capable variant; a GLRT framing yields a principled detection statistic. BOCPD is an option where a posterior over the change time is wanted.

### 4.3 Learned branch

**4.3.1 LightGBM on window features (interpretable challenger).**
Per-landing engineered features → regress the touchdown-time offset (and a quantile pair for uncertainty). Fast, competitive at this data scale, and its feature importances support the safety narrative.

**4.3.2 Sequence model (expected primary accuracy winner).**
A Temporal Convolutional Network or BiLSTM over the landing window, trained with **soft per-timestep labels** — a Gaussian bump centered on the QAR touchdown time rather than a single one-hot sample. Output P(touchdown) per timestep; take the **expected value** as `t_td` (gives sub-sample resolution) and the distribution width as uncertainty. Inputs are the physics-derived channels of §3.3 (hybrid input), the time-delta channel, and static context (aircraft type and source as embeddings).

**4.3.3 Hybrid residual (optional).**
Train the learned model to predict the **residual of a physics estimate** rather than the absolute time — captures systematic lever-arm/flare/latency bias while keeping a physical backbone.

### 4.4 Fusion (Module 5)
- Calibrated weighted blend / stacking of the per-estimator outputs, weights fit on grouped QAR validation.
- Per-estimator reliability can be conditioned on context (type, source, data-quality flags) so the ensemble down-weights an estimator where it is known to be weak (e.g., flare-crossing when geometric altitude is biased).
- Output: fused `t_td`, a predictive interval, and the contributing per-method estimates for traceability (NFR-1).

---

## 5. Touchdown Time → Position Mapping (Module 6)

1. Interpolate the **horizontal trajectory** at fused `t_td` (kinematic interpolation, §3.2).
2. This yields the **GNSS-antenna** position; apply the along-body **lever-arm correction** (pitch attitude at touchdown × antenna-to-gear distance) to get main-gear contact.
3. Project onto the runway centerline (from threshold + runway heading): report **along-runway distance from threshold** and **lateral offset**.
4. Sanity-bound against runway length; flag estimates beyond the runway end or before the threshold.

---

## 6. Validation Harness (Module 7)

- **Grouped splits** by tail / airport / runway (no random splitting — Errors §9.7).
- **Naive baseline** (first on-ground sample, and nearest-sample altitude crossing) computed for lift comparison.
- **Cross-source** evaluation (train source 1 → test source 2 and vice versa).
- **Stratified metrics** by type, source, airport/runway, approach-speed band.
- Report signed and absolute **distance error**, **time error**, full distributions, and **95th/99th-percentile long-side** tail.
- **Truth-alignment audit**: estimate and monitor the QAR↔ADS-B clock offset distribution.
- Per-method and ensemble leaderboards, plus calibration plots for the uncertainty estimates (do the stated intervals have the right coverage?).

---

## 7. Uncertainty Quantification

- LightGBM: quantile regression heads.
- Sequence model: width of the soft-label output distribution, optionally a **deep ensemble** (≈5 models) for epistemic uncertainty.
- Physics estimators: propagate fit covariance (knee breakpoint variance; IMM smoother covariance).
- Validate that nominal intervals achieve nominal coverage on held-out QAR (reliability/calibration check). Miscalibrated intervals are worse than none in a safety context.

---

## 8. Suggested Module / Repo Structure

```
tdz/
  io/            ingest, schema validation, QAR join
  timebase/      timestamp handling, resampling, kinematic interpolation
  signals/       smoothing, derivatives (SavGol/GP), feature construction
  geo/           projection to centerline, lever-arm correction, distance
  estimators/
    physics/     decel_knee, flare_crossing, imm
    changepoint/ jerk, cusum, pelt
    learned/     lightgbm, sequence (tcn/bilstm), hybrid_residual
  fusion/        calibration, stacking, reliability weighting
  validation/    splits (grouped), metrics, stratified reports, calibration
  config/        lever-arm tables, thresholds, method selection
  tests/         known-answer unit tests for geo/timebase/signals
```

- Common estimator interface: `estimate(flight) -> TDEstimate(t_td, sigma_t, diagnostics)`.
- Configuration-driven (NFR-3); every run records data/code/model/config versions (NFR-4).

---

## 9. Error Sources and Implementation Pitfalls

> This is the section to read twice. Most real-world error here is not in the model — it is in time, geometry, and leakage.

### 9.1 Asynchronous position/velocity timestamps
Position and velocity (source 1) are emitted at different times. **Naively merging them onto one row treats them as simultaneous and injects a position error of `velocity × Δt`** — at 130 kt and a 2 s offset that is ~130 m, comparable to the answer you're trying to estimate. Always carry both timestamps; interpolate with a kinematic model; prefer continuous-time consumption in the state-estimation branch.

### 9.2 QAR ↔ ADS-B clock offset (the silent bias)
A constant or drifting offset between the truth clock and the ADS-B clock **biases every calibrated and learned estimate by exactly that offset** and is invisible in internal cross-validation. Estimate the offset explicitly (e.g., align a robust common event), audit its distribution across flights, and treat any residual drift as label noise to bound. This is the highest-leverage correctness check in the whole system.

### 9.3 Differentiation noise at 4–5 s cadence
Jerk is the second derivative of groundspeed. Naive finite differencing of sparse, noisy speed makes jerk almost pure noise. Use Savitzky–Golay or GP-derivative estimates, and treat the jerk method as corroboration rather than a sole estimator. Validate derivative quality against QAR-derived accelerations on a sample.

### 9.4 Altitude pitfalls near the ground
- **Baro altitude** is QNH/transition-sensitive and can sit tens of feet off true height near the surface — do not use it for the absolute vertical crossing; use **geometric** altitude.
- **Geometric altitude can carry a static bias**; check it against known runway elevation per source and subtract before the crossing solve.
- **Fitting a straight line through the flare biases the crossing long** because the real profile flattens — model the flare explicitly.
- **Vertical rate is frequently null** and noisy — corroborate, never depend.

### 9.5 Peak deceleration ≠ touchdown
Maximum braking deceleration occurs several seconds **after** touchdown. Using the deceleration *peak* as the event biases late by tens to a couple hundred meters. Detect deceleration **onset** (knee / jerk onset), not magnitude.

### 9.6 On-ground flag latency
The air/ground bit transitions late and variably. Using it as the estimate produces a systematic long bias. Use it only as a coarse upper bracket on `t_td`.

### 9.7 Data leakage via random splits
The single most common way to ship an overoptimistic model. The same tail, airport, runway, or even flight context appearing in both train and test inflates metrics and hides domain shift. **Group splits by tail/airport/runway.** Also avoid window overlap leakage between train and test segments of the same flight.

### 9.8 Cross-source domain shift
The two ADS-B sources differ in fields, timestamping, noise, and update behavior. A model tuned on source 1 may silently degrade on source 2. Always evaluate cross-source; consider source as a feature/embedding; keep the physics anchor for the under-represented source.

### 9.9 Geometry and reference errors
- **Displaced thresholds:** distance must be measured from the **landing threshold**, not the paved runway start. A wrong reference is a fixed distance bias on every flight at that runway.
- **Lever-arm correction:** the antenna is above and offset from the main gear; omitting the height correction biases the vertical crossing, omitting the along-body/pitch correction biases the position. Both are aircraft-type dependent.
- **Projection sign/heading:** confirm runway heading direction and projection sign with known cases; a flipped sign turns long landings into short ones.
- **Geodesy:** use proper geodesic/ENU math near the runway, not naive lat/long Euclidean distance, especially at high latitudes.

### 9.10 Class imbalance and label smoothing
Touchdown is one instant among many samples. Hard one-hot labels make the learning problem needlessly sparse and brittle to label-time noise; use soft (Gaussian) labels and an expected-value readout.

### 9.11 Aircraft-type skew and rare strata
If a few types dominate the 40,500, rare types get little learned support and larger errors. Audit the distribution; lean on physics anchors and report per-type so thin strata are not hidden inside a good global average.

### 9.12 Tail blindness
Mean/RMSE can look excellent while the long-landing tail — the part that matters for overrun safety — is poor. Always report and gate on the 95th/99th-percentile long-side error, not just central tendency.

### 9.13 Uncertainty miscalibration
Reported confidence intervals that do not achieve their nominal coverage mislead downstream risk decisions. Validate calibration on held-out QAR and recalibrate (e.g., conformal / isotonic) if coverage is off.

### 9.14 Bounced / multiple touchdowns
Bounces produce more than one contact. Decide and document whether the system reports first contact; ensure the smoother/segmenter does not average across a bounce into a physically meaningless midpoint.

---

## 10. Recommended Build Order

1. **Pipeline + geometry + mapping** with unit tests (the parts that silently bias everything). Establish the QAR clock-offset audit.
2. **Physics decel-knee + change-point (PELT/CUSUM)** baselines → validated error floor, exposes data issues early.
3. **LightGBM** feature model → first strong, interpretable learned result.
4. **Sequence model (TCN/BiLSTM, soft labels)** → measure lift over baselines.
5. **Fusion + uncertainty calibration**, then full stratified / cross-source validation.

Each stage is independently shippable and the earlier stages remain the explainable anchors for the later ones.
