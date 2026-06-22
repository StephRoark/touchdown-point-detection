# Touchdown Point Inference from ADS-B

Infer where (and when) an aircraft touched down on the runway, using only ADS-B surveillance data plus runway geometry — and quantify how much to trust each estimate. The goal is landing-safety analysis: identifying long landings and high-speed touchdowns that drive runway-excursion and overrun risk.

Touchdown is never directly reported in ADS-B. This project reconstructs it from the trajectory, calibrates and validates the result against 40,500 QAR-matched flights, and reports a per-flight confidence interval rather than a bare number.

For the authoritative specification see **[requirements.md](.kiro/specs/touchdown-point-detection/requirements.md)** (what the system must do) and **[design.md](.kiro/specs/touchdown-point-detection/design.md)** (how it does it). This README is the orientation layer: the idea, the approach, and how to build and test it with data.

---

## Why this is hard

ADS-B was not designed to observe touchdown, and four properties of the data shape every design decision:

- **Coarse cadence (~4–5 s).** Touchdown almost always falls *between* samples. At ~140 kt the aircraft covers ~270 m in one update, so a "nearest sample" answer is structurally limited to several-hundred-metre error. Sub-sample estimation is mandatory.
- **Asynchronous messages.** On the primary source, position and velocity arrive with *different* timestamps. Merging them as if simultaneous injects a `velocity × Δt` position error (~130 m at 130 kt and 2 s) — comparable to the quantity being estimated.
- **Noisy / partial vertical data.** Barometric vertical rate is often null; barometric altitude is QNH-sensitive near the ground; geometric (GNSS) altitude is in a different vertical datum (HAE) than published runway elevations (MSL).
- **A delayed on-ground flag.** The air/ground bit transitions late and variably, so it can only bound the answer, never be the answer.

## The approach

**Estimate a *time*, then map it to a *place*.** Every method targets a sub-sample touchdown time `t_td`. The horizontal position is obtained afterward by interpolating the trajectory at `t_td` and projecting onto the runway centerline. This separates *detection* (when) from *geolocation* (where) so all methods share one mapping and one validation back-end.

**Physics for trust, learning for accuracy.** With 40,500 labels a neural sequence model is expected to be the most accurate estimator, but a safety case needs explainability. So the design always pairs the learned model with an interpretable physical anchor, and feeds physics-derived signals *into* the learned model so it refines physics rather than rediscovering it.

**Multiple estimators, calibrated fusion.** Different methods fail on different landings (null vertical rate, geometric-altitude bias, long/balked flares, source quirks). Several independent estimators are fused with weights calibrated on QAR truth, which reduces variance and — most importantly for overrun analysis — the long-landing tail.

The estimator families:

- **Physics / state-estimation** — a deceleration-knee fit (segmented regression on raw groundspeed), a glideslope-plus-flare vertical crossing, and an IMM (interacting multiple model) filter whose flight→ground mode crossover marks touchdown.
- **Change-point** — CUSUM / PELT / GLRT and a jerk-onset detector on the deceleration signal.
- **Learned** — LightGBM on engineered window features (the interpretable challenger) and a TCN/BiLSTM sequence model trained with soft Gaussian labels (the expected accuracy winner).

A handful of decisions matter more than the model choice and are easy to get wrong; they are the difference between a credible system and a silently biased one:

- Preserve asynchronous timestamps; interpolate position with a kinematic (dead-reckoning) model, never a naive merge.
- Unify the vertical datum: geoid-correct the runway elevation to HAE before any altitude crossing, and keep that deterministic correction separate from any empirical sensor-bias estimate.
- Fit the vertical crossing over an extended region (~200–300 ft down), because at this cadence only one or two samples fall below 50 ft.
- Pitch is not in ADS-B; resolve the lever arm with a per-type *assumed* nominal touchdown pitch.
- Distance truth is clock-independent (it comes from the QAR touchdown lat/long), so clock alignment is needed only for *time* labels — confining clock risk to the time domain.
- Run a coarse, on-ground-flag-independent bracket first, then refine; classify go-arounds and touch-and-goes so the system never forces a touchdown where none occurred.

## Repository layout

The design specifies a 7-module pipeline. Each estimator emits the same contract `(t_td, sigma_t, diagnostics)`, so fusion, mapping, and validation are method-agnostic.

```
tdz/
  io/            ingest, schema validation, QAR join, source-capability gating, datum unification
  bracket/       trajectory classification (landing/go-around/touch-and-go), coarse touchdown bracket
  timebase/      async-timestamp handling, kinematic interpolation, optional common-grid resample
  signals/       segmented regression (primary), SavGol/GP derivatives (corroboration), features
  geo/           geodesy, geoid model, lever-arm correction, centerline projection, wrong-runway gate
  estimators/
    physics/     decel_knee, flare_crossing, imm
    changepoint/ jerk, cusum, pelt, glrt
    learned/     lightgbm, sequence (tcn/bilstm), hybrid_residual
  fusion/        calibration, stacking, reliability weighting
  validation/    grouped splits, stratified metrics, calibration, cross-source evaluation
  config/        lever-arm tables, geoid model, source descriptors, thresholds, method selection
  tests/         property-based + example-based unit tests, integration, validation-vs-QAR
config/           YAML configs (see design.md §Configuration Schema)
```

## Data you need

| Input | Used for | Notes |
|------|----------|-------|
| ADS-B trajectories | The estimate | Position (lat/long), groundspeed, track, geometric + barometric altitude, baro vertical rate (often null), on-ground flag, per-message timestamps. Two sources supported via a capability descriptor. |
| Runway reference | Position mapping | Threshold lat/long, heading, **elevation with an explicit HAE/MSL datum tag**, length, width. Use the *landing* (displaced) threshold. |
| Aircraft type + lever-arm table | Per-type correction | ICAO type → antenna vertical/longitudinal offset, nominal touchdown pitch, aircraft class. Missing types fall back to a class-median default. |
| QAR truth (40,500 flights) | Calibration + validation | Touchdown timestamp **and** lat/long. The lat/long gives clock-independent distance truth; the timestamp (after clock alignment) gives time labels. |

A note on the second source (assumed to be FlightRadar24): it is currently treated as **barometric-only and provider-interpolated** until confirmed. That assumption lives entirely in the `sources` block of the config, so geometric/vertical estimators are auto-disabled for it and its samples are not treated as independent observations. Flip two flags once you confirm the real characteristics.

## Implementation plan

Build in stages so every layer is validated before the next is added. Each stage is independently shippable and the earlier (physics) stages remain the explainable anchors for the later (learned) ones.

1. **Geometry, datum, and mapping — with unit tests first.** These are the parts that *silently bias everything*: centerline projection (geodesic/ENU, not Euclidean), geoid correction, lever-arm + pitch correction, kinematic interpolation. Establish the QAR clock-offset audit here too. Get this provably right before any estimator exists.
2. **Coarse bracket + trajectory classification.** Flag-independent touchdown bracketing; reject go-arounds, tag touch-and-goes.
3. **Physics + change-point baselines.** Deceleration-knee and PELT/CUSUM give a validated error floor and surface data-quality problems early.
4. **LightGBM feature model.** First strong, interpretable learned result.
5. **Sequence model (TCN/BiLSTM, soft labels).** Measure the lift over the baselines.
6. **Fusion + uncertainty calibration**, then the full stratified, cross-source validation report.

Suggested stack: Python 3.11+, NumPy/SciPy/pandas, `pyproj` (geodesy + geoid), `ruptures` (PELT/CUSUM), `filterpy` (Kalman/IMM), `scikit-learn`, `lightgbm`, PyTorch (sequence model), `pytest` + `hypothesis` (tests).

## Testing with data

Testing happens at three levels, deliberately separated so a passing unit test never depends on an unverified model.

**1. Correctness tests (no QAR needed).** The design enumerates correctness properties (P1–P23). Each maps to a Hypothesis property-based test (≥100 randomized iterations) plus targeted known-answer cases. The high-value ones:

- Runway projection round-trips to within 0.1 m; geodesic vs Euclidean checked at high latitude.
- Kinematic interpolation introduces < 30 ft of position error for known timestamp offsets at 120–150 kt (the async-merge trap).
- Geoid correction: a synthetic crossing with a known undulation recovers the correct height; MSL and HAE inputs agree after correction.
- Lever-arm shift equals `longitudinal·cosθ + vertical·sinθ` at the nominal pitch.
- On-ground flag never exceeds the bracket; go-around → no touchdown; bounce → first-contact, not an average.
- Grouped split leaks no tail across train/test; config schema rejects bad values.

Synthesize landings (constant glideslope + flare + constant-deceleration rollout) at 4–5 s cadence with injected gaps, duplicate timestamps, null vertical rate, and impossible accelerations to exercise the quality gates without touching real data.

**2. Validation against QAR truth (the accuracy story).** This is where the 40,500 flights come in.

- **Distance is the headline metric and is clock-independent** — compute along-runway truth directly from the QAR touchdown lat/long, no clock alignment required. Report *signed* distance error (so long/short bias is visible), absolute error, RMSE, median, IQR, and the **95th/99th-percentile long-side** error, because overrun risk lives in the tail.
- **Time error** is the cleaner estimator diagnostic and *does* require clock alignment: estimate the QAR↔ADS-B offset by cross-correlating the overlapping groundspeed/position series (never align on touchdown itself), and exclude drifting/large-offset flights from the *time* metric only.
- **Splits must prevent leakage.** Use a tail-grouped train/calibration/test split for the headline numbers; report held-out-airport and held-out-runway evaluations *separately* as geographic-generalization stress tests. Random splitting is prohibited — it inflates everything.
- **Always beat the baseline.** Compute the naive "first on-ground sample" estimate on the same test set and report side-by-side; the system must materially outperform it.
- **Cross-source.** You have two ADS-B providers (the primary feed and the assumed-FlightRadar24 feed), which differ in fields, timestamping, noise, and sampling. Train the learned models on one provider and test on the other (both directions), and report the accuracy drop versus same-provider testing — this measures *provider* generalization and is where real-world error hides, distinct from the held-out-airport/runway tests above (which measure *geographic* generalization). Note the two feeds are not interchangeable: the geometric/vertical estimators are auto-disabled for the source lacking geometric altitude, so the ensemble itself differs across sources. For the comparison to isolate the source effect, use flights present in both feeds (or matched aircraft/airport distributions) so it isn't confounded by a different flight mix.
- **Calibration.** Verify the 90% intervals actually cover ~90% (target 85–95%) on the held-out calibration split; conformalize if they don't. Miscalibrated intervals are worse than none in a safety context.
- **Stratify everything** by aircraft type, source, airport, and approach-speed band (≥30 flights per reported stratum).
- **Accuracy targets are provisional** until a baseline run characterizes the irreducible, cadence-limited error floor; until then they are reporting targets, not pass/fail gates.

**3. Integration tests.** End-to-end on a synthetic flight (verify the full output record); both source formats (async vs co-timed) on identical trajectories; a small QAR subset through the whole validation harness.

A practical first milestone: run stages 1–3 on a held-out QAR slice, produce the distance-error distribution and the baseline comparison, and confirm the cadence-limited floor — that single report tells you whether the physics alone is already close, and how much room the learned models have to add.

## Status and open items

- **FR24 (second source) provenance** — altitude type and raw-vs-interpolated samples unconfirmed; encoded as a config-gated assumption.
- **Accuracy targets** — provisional pending the cadence-floor baseline.
- **Lever-arm default** — implemented as class-median + low-confidence + widened CI (not a worst-case bias), per the design rationale.

## See also

- **[requirements.md](.kiro/specs/touchdown-point-detection/requirements.md)** — 21 EARS-style requirements with acceptance criteria.
- **[design.md](.kiro/specs/touchdown-point-detection/design.md)** — architecture, data models, correctness properties, error handling, and testing strategy.
- **[tasks.md](.kiro/specs/touchdown-point-detection/tasks.md)** — the staged, test-driven implementation plan.
