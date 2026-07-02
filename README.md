# Touchdown Point Inference from ADS-B

Infer where (and when) an aircraft touched down on the runway, using only ADS-B surveillance data plus runway geometry — and quantify how much to trust each estimate. The goal is landing-safety analysis: identifying long landings and high-speed touchdowns that drive runway-excursion and overrun risk.

Touchdown is never directly reported in ADS-B. This project reconstructs it from the trajectory, calibrates and validates the result against 40,500 QAR-matched flights, and reports a per-flight confidence interval rather than a bare number.

For the authoritative specification see **[requirements.md](.kiro/specs/touchdown-point-detection/requirements.md)** (what the system must do) and **[design.md](.kiro/specs/touchdown-point-detection/design.md)** (how it does it). This README is the orientation layer: the idea, the approach, and how to build and test it with data.

---

## The story version (start here if none of the words above meant anything)

When an airplane lands, the spot where its wheels first touch the runway really matters. A runway is long, but not endless — if a plane touches down too far along it, or comes in too fast, there may not be enough pavement left to stop. That's how planes end up sliding off the end of runways, which you occasionally see on the news.

Airlines would love to look back at thousands of old landings and ask: *which ones touched down too far, or too fast?* Those are the near-misses worth learning from before one becomes an accident. The trouble is, nobody stands at the runway with a clipboard writing down where each plane's wheels touched.

Here's what we *do* have. Every airliner constantly calls out over the radio, "here I am, this is how fast I'm going" — automatically, about once every four or five seconds. Those messages get recorded. So for any landing, we have a trail of breadcrumbs leading down to the runway.

But there's a catch: four seconds is a long time for a landing airplane. Between one radio call and the next, the plane travels about the length of *three football fields*. And the moment the wheels touch almost always happens *between* two calls — never exactly on one. So you can't just look up the answer. It's like trying to figure out the exact moment a pot of soup started boiling when you only peeked at the stove once every few minutes: you have to *work it out* from the clues on either side.

That's what this project does. It works out the touchdown moment from the clues — the plane slows down suddenly once its wheels are rolling and the brakes bite; its height tapers down to the runway in a predictable curve — and it uses several different ways of guessing at once, then blends them, because each way stumbles on different landings.

Two more things make it trustworthy rather than just clever:

1. **We have an answer key.** For 40,500 landings, the airplane's own onboard recorder tells us where the wheels *really* touched. We grade every guessing method against those answers before trusting it on landings where there is no recorder.
2. **Every answer admits how sure it is.** The system never just says "the plane touched down 1,500 feet in." It says "1,500 feet, give or take 200." For safety work, an honest *give-or-take* is as important as the number — a confident wrong answer is the most dangerous thing this kind of system can produce.

That's the whole idea: use the breadcrumb trail every plane already broadcasts to find the exact touchdown spot, check ourselves against the flights where the true answer is known, and always say how confident we are — so safety people can find the risky landings hiding in a mountain of ordinary ones.

## The executive summary

Runway excursions — aircraft departing the runway surface on landing — are consistently among the top categories of serious commercial-aviation accidents, and the two leading precursors are touching down too far along the runway and touching down too fast. Today those precursors can only be measured on flights that carry and share flight-data-recorder (QAR) downloads: a minority of operations, unevenly distributed across fleets and operators.

This system measures both precursors from **ADS-B surveillance data, which exists for essentially every flight**, at no additional cost of collection. Because touchdown is never directly reported in ADS-B, the system reconstructs it: multiple independent estimation methods (physics-based, statistical, and machine-learned) are combined, and every estimate carries a calibrated confidence interval, so analysts know not just *where* a landing touched down but *how much to trust that answer*. A library of 40,500 flights with matched recorder data serves as ground truth for calibration and for graded, audited accuracy claims — including an interpretable physics estimate alongside every machine-learned one, for regulator-facing explainability.

**Status:** the full pipeline — ingestion, quality control, seven estimation methods, fusion, uncertainty calibration, and the validation harness — is built and passes a 460-test suite. The remaining step is empirical: running the recorder-matched corpus through the system to confirm the working accuracy target (250 ft RMSE on touchdown point) is achievable at the 4–5-second ADS-B reporting cadence. The accuracy targets are deliberately held as provisional until that baseline run ratifies them.

**The decision this enables:** fleet-wide, continuous monitoring of landing-overrun risk across all operations — not just the recorder-equipped subset — with statistically honest per-landing confidence.

## Slide-deck talking points

**Slide — the problem.**
- Runway overruns are a leading serious-accident category; long and fast touchdowns are the measurable precursors.
- Today we can only measure touchdown on flights with recorder (QAR) data — a fraction of operations.

**Slide — the opportunity.**
- Every airliner already broadcasts position and speed every 4–5 seconds (ADS-B); it's recorded fleet-wide.
- If touchdown can be inferred from ADS-B, overrun-risk monitoring extends to essentially *all* landings.

**Slide — why it's hard.**
- At landing speed, a plane covers ~900 ft between ADS-B reports; touchdown always falls *between* reports.
- The one signal that sounds helpful (the aircraft's "on-ground" flag) flips late and inconsistently — usable only as a sanity bound, never as the answer.
- Position and speed messages aren't even timestamped together; naïvely merging them injects errors as large as the quantity being measured.

**Slide — the approach.**
- Estimate the *time* of touchdown to sub-second precision, then map time → runway position.
- Seven independent estimators across three families — physics (deceleration knee, descent-curve crossing, tracking filter), statistical change-point detection, and machine learning trained on recorder-matched flights.
- A calibrated fusion blends them, down-weighting whichever methods look unreliable on each specific landing.

**Slide — why to trust it.**
- 40,500 recorder-matched flights are the answer key: every method is graded against known truth, with leakage-controlled train/test splits.
- Every output carries a calibrated confidence interval — verified so that "90% confident" empirically means ~90%.
- Every machine-learned estimate ships with an interpretable physics estimate alongside it, for explainability to regulators.
- The system says "no estimate" rather than guessing when data quality is insufficient, with a machine-readable reason.

**Slide — status and next step.**
- Pipeline complete: ingestion → quality gates → estimators → fusion → calibrated uncertainty → validation harness; 460 automated tests.
- Next: run the recorder-matched corpus to establish the accuracy floor imposed by the 4–5 s data cadence and ratify the provisional 250 ft RMSE target.

## For technical peer reviewers

The fastest path to a substantive review:

1. **Read the specs in order.** [requirements.md](.kiro/specs/touchdown-point-detection/requirements.md) (21 EARS-style requirements — the *what*), then [design.md](.kiro/specs/touchdown-point-detection/design.md) (architecture, data models, 23 correctness properties — the *how*). The "Key Technical Decisions" table in the design doc is the 5-minute version of every load-bearing choice and its rationale.
2. **Know the four data pathologies everything is designed around.** Coarse cadence (4–5 s, so touchdown falls between samples); asynchronous position/velocity timestamps (naïve merging injects ~130 m at approach speed); unreliable vertical data (baro is QNH-sensitive, geometric altitude is in HAE while runway elevations are MSL — a tens-of-meters datum trap); and a late, variable on-ground flag (upper bound only, never a measurement). Most of the design is a considered response to one of these.
3. **The architecture is a 7-module DAG** (ingest/QA → classification/bracketing → timebase → signals → estimators → fusion → mapping, with validation alongside). Every estimator emits the same contract `(t_td, sigma_t, diagnostics)`, so fusion, mapping, and validation are method-agnostic. Module-to-package mapping is in the design doc and mirrored by the repository layout below.

**The claims most worth scrutinizing** (and where they're enforced):

- *No leakage in the headline metrics* — tail-grouped train/calibration/test splits, with held-out-airport and held-out-runway evaluations reported separately rather than intersected (`tdz/validation/splits.py`, Property 10).
- *Distance truth is clock-independent* — along-runway truth is projected from the QAR touchdown lat/long, so QAR↔ADS-B clock offset can corrupt only time-domain labels, not the primary distance metric (`tdz/validation/clock_alignment.py`, Req 12.10/19.3).
- *Honest intervals* — model-based intervals are recalibrated by normalized split conformal on a partition disjoint from both training and test (`tdz/uncertainty/conformal.py`, Req 4.3/4.4: 90% intervals must land in 85–95% empirical coverage).
- *Datum discipline* — MSL runway elevations are geoid-corrected to HAE deterministically, kept separate from any empirical sensor-bias estimate (`tdz/geo/datum.py`, `tdz/estimators/physics/flare_crossing.py`, Req 11.2/17.3).
- *The async-merge trap* — position is dead-reckoned to query times using the velocity stream; the two timebases are never merged (`tdz/timebase/interpolation.py`, Property 3: <30 ft error bound, enforced by test).

**Known assumptions and open items to challenge:**

- The FR24 source is *assumed* barometric-only and provider-interpolated (config-gated, unconfirmed) — vertical estimators are disabled for it.
- Touchdown pitch is *assumed* per aircraft type (pitch is not in ADS-B); missing lever-arm entries fall back to a class median with widened CIs rather than a worst-case bias — challenge whether that default is right for your fleet mix.
- The 250 ft RMSE target is provisional until the cadence-limited error floor is characterized on real data (Req 13.0); all results so far are on synthetic trajectories.
- QAR truth is treated as exact; its own touchdown-position uncertainty has not yet been characterized and would inflate apparent error.

**Verify it yourself:** `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]" && .venv/bin/python -m pytest` — 460 tests, ~20 s, no data or network needed (one geoid test requires the EGM2008 grid, see Environment setup). Property-based tests (Hypothesis, ≥100 randomized cases each) cover the 23 correctness properties; the mapping from property to test is tagged in the test docstrings.

## The big picture, in plain language

**The one-sentence version:** an airplane's transponder reports where it is every few seconds; we use that breadcrumb trail to figure out the exact spot on the runway where its wheels first touched, how fast it was going, and how sure we are — so safety analysts can spot landings that touched down too far down the runway or too fast.

**An everyday analogy.** Imagine a car's phone reports its GPS location once every 4–5 seconds. You want to know the exact point where the car crossed a finish line painted on the road. The car covers the length of a football field between two reports, and the finish-line crossing almost always happens *between* two pings — never exactly on one. So you can't just pick the closest ping; you have to reconstruct the motion and infer the in-between moment. That is essentially this problem, with an airplane, a runway threshold instead of a finish line, and safety stakes attached.

### What we're measuring, and why it matters

- **Touchdown point** — the distance from the start of the runway (the "threshold") to where the main wheels first hit. Touch down too far along the runway (a "long landing") and there may not be enough pavement left to stop, which risks a **runway overrun**.
- **Touchdown speed** — how fast the aircraft was moving when it landed. Faster touchdowns need more stopping distance, which also raises overrun risk.
- **A confidence interval** — instead of a single number ("1,500 ft past the threshold"), every estimate comes with an honest range ("1,500 ft, give or take 200"). In safety work, knowing *how much to trust* a number is as important as the number itself.

These outputs let analysts comb through tens of thousands of historical landings and flag the risky ones for review.

### Where the data comes from

- **ADS-B** (Automatic Dependent Surveillance–Broadcast): airplanes continuously broadcast their position, speed, and altitude. Ground stations and satellites record it. It's widely available, but it was built for air-traffic awareness, not for pinpointing touchdown — hence the difficulty. Updates arrive only every ~4–5 seconds.
- **QAR** (Quick Access Recorder): an onboard recorder that captures what really happened, including the true touchdown time and location. We have 40,500 flights with matched QAR data. We use it as the "answer key" to calibrate and grade the system — but the finished system runs on ADS-B alone, because most flights don't come with QAR data.
- **Runway and aircraft reference data**: where each runway's threshold sits, its heading, length, and width; and per-aircraft-type details like where the GPS antenna sits relative to the wheels.

### Key terms in one place

| Term | Plain meaning |
|------|---------------|
| **Threshold** | The line where the usable landing runway begins; we measure touchdown distance from here. |
| **Flare** | The gentle nose-up maneuver pilots make just before landing to slow the descent; it curves the altitude path right before touchdown. |
| **On-ground flag** | A bit in the data that flips from "airborne" to "on ground." It flips *late and unreliably*, so we use it only as a rough upper bound, never as the answer. |
| **Lever arm** | The GPS antenna sits on top of the fuselage, not at the wheels. That offset (and the aircraft's nose-up angle) must be corrected for, or the touchdown point is wrong by tens of feet. |
| **HAE vs MSL** | Two different "zero points" for altitude — height above the GPS ellipsoid (HAE) vs height above sea level (MSL). Runway elevations and GPS altitudes often use different ones; mixing them adds a tens-of-meters error. |
| **Go-around / touch-and-go** | A go-around is an aborted landing (no touchdown); a touch-and-go briefly touches then takes off again. The system must recognize these so it doesn't report a touchdown that never happened. |
| **Sub-sample estimation** | Pinpointing an event to finer than the 4–5 second data spacing — the core trick of the whole project. |

---

## Why this is hard

In short: the data is coarse, slightly misaligned in time, partly missing, and the one obvious signal (the on-ground flag) is unreliable. The four properties below shape every design decision.

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

## How a single landing flows through the system

Here is the journey of one flight's data, end to end, in plain terms. Each numbered step matches a module in the code layout below.

1. **Ingest & quality-check.** Read the raw ADS-B for the flight, attach the runway and aircraft-type reference data, and clean the data: drop duplicate or physically impossible samples (e.g. an implied 2-g deceleration), note which signals are missing, and put the runway elevation into the same altitude "zero point" (datum) as the GPS altitude. If the data is too sparse or broken, the flight is flagged and no estimate is forced.
2. **Classify and bracket.** Decide whether this was a normal landing, a go-around (no touchdown), or a touch-and-go. For real landings, draw a rough time window — the "bracket" — that is very likely to contain the touchdown. This solves a chicken-and-egg problem: later quality checks need to look "near touchdown," but we don't know touchdown yet, so we use this rough window as the stand-in.
3. **Build a clean timeline.** Because position and speed can arrive with different timestamps, the system never naively merges them. It reconstructs the motion (dead-reckoning from speed) so it can ask "where was the aircraft at *exactly* this instant," and builds smoothed speed/deceleration signals.
4. **Run several independent estimators.** Multiple methods each estimate the touchdown *time*:
   - **Physics methods** read it from the kinematics — where the deceleration sharply changes (the "knee"), where the altitude path crosses the runway, or where a tracking filter switches from "flying" to "rolling."
   - **Change-point methods** statistically detect the moment the motion changes regime.
   - **Learned methods** are trained on the 40,500 QAR examples to predict touchdown from the data patterns.
   Each estimator reports its time, its uncertainty, and supporting diagnostics in the same standard format.
5. **Fuse them.** Combine the estimators into one answer, weighting each by how reliable it looks for this flight and down-weighting or dropping ones that disagree or report low confidence. The result is a single touchdown time plus a calibrated confidence interval.
6. **Map time to place.** Take the fused touchdown time, find the aircraft's position at that instant, correct for the antenna-to-wheels offset, and project onto the runway centerline to get the distance from the threshold (plus sideways offset and touchdown speed). Convert to feet/knots only here, at the very end.
7. **Validate against truth.** Separately, compare the system's answers to the QAR "answer key" across many flights, measure the error, confirm the confidence intervals are honest, and check it beats a naive baseline — all with careful data splitting so the scores aren't inflated.

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

The system is built in stages, deliberately ordered so that the parts that can *silently corrupt every result* are built and proven correct first, before any estimator depends on them. Each stage is independently shippable and testable, and the earlier physics stages stay in place as the explainable anchor for the later learned stages. Detailed, checkbox-level tasks live in **[tasks.md](.kiro/specs/touchdown-point-detection/tasks.md)**.

1. **Geometry, datum, and mapping — tested first.** The unglamorous foundation: projecting a position onto the runway centerline (using proper curved-earth geodesy, not flat-earth shortcuts), correcting the runway elevation into the right altitude datum, correcting for the antenna-to-wheels offset, and reconstructing position between samples. A small error here biases *every* landing in the same direction without ever throwing an error, so each piece ships with its own correctness tests before anything is built on top of it. *What "done" looks like: projection round-trips to within 0.1 m, and bad runway data is rejected cleanly.*
2. **Coarse bracket + trajectory classification.** Decide landing vs go-around vs touch-and-go, and draw the rough touchdown time window every later check relies on. *Done: go-arounds produce no touchdown; a bounce is anchored to the first contact, not an average.*
3. **Physics + change-point baselines.** The deceleration-knee and statistical change-point estimators. These are interpretable and give a trustworthy accuracy floor, and they surface real data-quality problems early. *Done: a first end-to-end estimate exists and beats the naive baseline.*
4. **LightGBM feature model.** The first machine-learned estimator — gradient-boosted trees on engineered features. Strong, fast, and still interpretable (you can see which features mattered). *Done: measurable improvement over the physics baseline.*
5. **Sequence model (TCN/BiLSTM).** A deep model over the whole landing window, trained with soft labels, expected to be the most accurate. *Done: its lift over the simpler models is quantified.*
6. **Fusion + uncertainty calibration, then full validation.** Blend the estimators, make the confidence intervals honest (so a "90%" interval really covers ~90%), and produce the full stratified, cross-source accuracy report. *Done: calibrated intervals and a complete validation report against QAR truth.*

Suggested stack: Python 3.11+, NumPy/SciPy/pandas, `pyproj` (geodesy + geoid), `ruptures` (PELT/CUSUM), `filterpy` (Kalman/IMM), `scikit-learn`, `lightgbm`, PyTorch (sequence model), `pytest` + `hypothesis` (tests).

### Progress so far

- ✅ **Stage 0 — Foundations.** Project scaffolding, shared data models, and a fully validated, configuration-driven settings module (every tunable lives in YAML, not in code).
- ✅ **Stage 1 — Geometry, datum, and mapping.** Geodesic runway-centerline projection + reference validation, geoid (MSL→HAE) datum unification, the pitch-resolved lever-arm correction, and the wrong-runway / out-of-bounds gates.
- ✅ **Stage 2 — Timebase, ingest/QA, bracketing.** Async-timestamp-preserving kinematic interpolation, dual-source ingest with capability gating, QA/quality gates, and flag-independent trajectory classification + coarse bracket.
- ✅ **Stage 3 — Physics + change-point baselines.** Decel-knee, flare-crossing, and IMM physics estimators; PELT/CUSUM/GLRT/jerk-onset change-point estimators; and a stage-1–3 baseline run that beats the naive first-on-ground strawman.
- ✅ **Stage 4 — Learned estimators.** The LightGBM window-feature estimator, the TCN/BiLSTM sequence model (soft Gaussian labels, optional deep ensemble), the hybrid residual model, and the rare-type physics fallback.
- ✅ **Stage 5 — Fusion + uncertainty.** The calibrated fusion ensemble (inverse-variance blend / stacking with gating and disagreement flags), split-conformal interval calibration, gap-proportional CI widening, and the time→position mapping / output-record assembly.
- ✅ **Stage 6 — Validation, reproducibility, reporting.** QAR clock alignment (cross-correlation, drift detection), tail-grouped + held-out-airport/runway splits, stratified metrics with the naive-baseline comparison, coverage assessment, provenance stamping, and the first-milestone report (all 24 tasks in tasks.md complete).

The suite currently stands at 450+ passing tests (1 skipped — a geoid-grid test that needs the optional EGM2008 grid). Raw input schemas for both the ADS-B timeseries and the QAR truth data are defined and feed the ingest layer. **The remaining work is empirical, not structural:** connect real ADS-B/QAR data files to the ingest layer, run the milestone report on a real QAR slice to characterize the cadence-limited error floor, and ratify the provisional accuracy targets.

## Environment setup

The project targets **Python 3.11+**. Create a virtual environment and install the package with its dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

`.venv/` is git-ignored. Run the test suite with `.venv/bin/python -m pytest`.

### System prerequisite: OpenMP (for LightGBM)

`lightgbm` (the learned window-feature estimator) loads the OpenMP runtime `libomp` at import time, which is **not** a pip package — it must be installed at the system level, or `import lightgbm` will fail with a missing-`libomp.dylib` error.

- **macOS (Homebrew):** `brew install libomp`. If Homebrew itself reports the macOS version as unsupported, run `brew update` first (a stale Homebrew predating your OS is the usual cause). On Intel Macs this lands at `/usr/local/opt/libomp`; on Apple Silicon at `/opt/homebrew/opt/libomp`. `libomp` is keg-only, which is expected — LightGBM finds it via the `opt` symlink. Do **not** hand-copy `libomp.dylib` into `/usr/local/opt/libomp` (a real directory there blocks the proper Homebrew install).
- **Apple Silicon migrated from an Intel Mac:** a Migration-Assistant-carried Homebrew at `/usr/local` provides only x86_64 `libomp`, which the arm64 LightGBM wheel cannot load (`incompatible architecture`). Install native arm64 Homebrew (at `/opt/homebrew`) and `brew install libomp` there. Interim workaround: torch's wheel bundles an arm64 `libomp`, so `DYLD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/torch/lib .venv/bin/python -m pytest` runs the suite. Likewise, a venv copied from an Intel Mac contains x86_64 wheels and must be recreated (`rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`).
- **Linux:** install your distro's OpenMP runtime (e.g. `libgomp1` on Debian/Ubuntu), or use the conda-forge `lightgbm` build which bundles it.
- **conda (any OS):** `conda install -c conda-forge lightgbm` pulls in the OpenMP runtime automatically.

### PyTorch (for the sequence model)

The TCN/BiLSTM sequence estimator (Stage 4) needs **PyTorch**. A CPU build is sufficient for development and tests; install the wheel appropriate for your platform from the [official selector](https://pytorch.org/get-started/locally/) (e.g. `pip install torch`). It is a large download; the physics, change-point, and LightGBM estimators do not require it.

#### Local development on Intel macOS (x86_64) — torch/NumPy bridge

This is a **local-dev-only** caveat; it does **not** affect the production target (Azure Databricks, Linux x86_64).

Apple dropped Intel-Mac PyTorch wheels after **torch 2.2.2**, which is therefore the newest installable build on an Intel Mac. That build was compiled against the NumPy **1.x** C-ABI, so under NumPy **2.x** its `torch.from_numpy` bridge fails to initialize (`_ARRAY_API not found`) and raises. The sequence model works around this with a small `_to_tensor` shim (`tdz/estimators/learned/sequence_model.py`) that falls back to a list round-trip when the fast path is unavailable — correct, just slower, and exercised **only** on this platform.

Notes:

- **Do not** add a project-wide `numpy<2` pin on account of this. The constraint is specific to the Intel-Mac dev box; capping NumPy would also collide with `scipy` (which requires `numpy>=2.0.0`).
- On Linux/Databricks, torch 2.3+ supports NumPy 2.x and the zero-copy fast path is always taken — the shim's fallback branch never runs there, and it is not a pattern to copy into production code.
- If you prefer the native bridge locally, pin a NumPy-1.x-compatible stack in the venv only (e.g. `numpy<2` **with** a `scipy<1.16`), but this is optional and should stay out of `pyproject.toml`.

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
- **Runway length/displacement input units** — `runway_length`, `threshold_diplacement_length` [sic], `opposite_displacement_length`, and `destination_runway_length` are *assumed meters* in the raw input schema (`tdz/io/raw_schema.py`, marked "TODO: confirm m vs ft"). Confirm with the data provider before the first real-data run: if the data is actually feet, runways over 6,000 (ft) are rejected loudly by the 0–6000 m validation bound, but shorter runways would pass silently with a 3.28× geometry error. (Output distances are always feet, converted once at the output boundary; internal computation is SI meters.)
- **Accuracy targets** — provisional pending the cadence-floor baseline.
- **Lever-arm default** — implemented as class-median + low-confidence + widened CI (not a worst-case bias), per the design rationale.

## See also

- **[requirements.md](.kiro/specs/touchdown-point-detection/requirements.md)** — 21 EARS-style requirements with acceptance criteria.
- **[design.md](.kiro/specs/touchdown-point-detection/design.md)** — architecture, data models, correctness properties, error handling, and testing strategy.
- **[tasks.md](.kiro/specs/touchdown-point-detection/tasks.md)** — the staged, test-driven implementation plan.
