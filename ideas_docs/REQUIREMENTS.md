# Touchdown Point Inference from ADS-B — Requirements

**Project:** Physics- and ML-based inference of aircraft touchdown point from ADS-B surveillance data
**Domain:** Aerospace safety analytics (landing performance, long-landing / overrun risk)
**Status:** Draft v1.0
**Validation truth source:** 40,500 QAR-matched flights

---

## 1. Purpose and Scope

### 1.1 Purpose
Touchdown point is not directly observed in ADS-B. This system infers, for each landing, the **touchdown time** and the corresponding **along-runway touchdown position** (distance from the landing threshold), using ADS-B kinematic data and known runway geometry. Estimates support safety analysis of landing performance — in particular long-landing and runway-overrun risk.

### 1.2 In Scope
- Estimation of touchdown **time** to sub-sample resolution from ADS-B time series.
- Mapping of touchdown time to an **along-runway distance** from the threshold and a lateral offset from centerline.
- Per-estimate **uncertainty** (confidence interval).
- A reproducible **validation harness** against QAR truth, with stratified error reporting.
- Support for **two ADS-B sources** with differing message structures and timestamping.
- Support for **multiple aircraft types**.

### 1.3 Out of Scope
- Real-time / online operational use (initial system is offline / post-hoc analysis). Online operation is a possible later extension.
- Detection of events other than touchdown (e.g., rotation, liftoff, taxi) — though the architecture should not preclude them.
- Bounce / multiple-touchdown decomposition beyond reporting the **first** main-gear contact (see Open Questions).
- Runway condition, braking-action, or friction estimation.
- Source data acquisition / ADS-B decoding (assumed upstream).

---

## 2. Definitions and Glossary

| Term | Definition |
|------|------------|
| Touchdown (TD) | First sustained main-gear contact with the runway surface. Reference truth taken from QAR weight-on-wheels / radio-altimeter = 0. |
| Touchdown time `t_td` | The time of TD on the ADS-B clock, estimated to sub-sample resolution. |
| Touchdown point | The horizontal position at `t_td`, expressed as along-runway distance from threshold and lateral offset from centerline. |
| Threshold | Beginning of the runway usable for landing; reference point for along-runway distance. Provided as lat/long. |
| ADS-B | Automatic Dependent Surveillance–Broadcast. Position and velocity arrive in separate message types with separate emission times. |
| Geometric altitude | GNSS-derived height (above WGS-84 ellipsoid). Preferred for absolute height near ground. |
| Barometric altitude | Pressure altitude; subject to QNH / pressure-setting error near the surface. |
| On-ground flag | ADS-B air/ground status bit; known to transition with delay. |
| QAR | Quick Access Recorder — onboard flight data used as ground-truth touchdown reference. |
| Update cadence | ADS-B sample interval, here ~4–5 s. |
| Antenna lever arm | Offset between the GNSS antenna and the main landing gear (height and along-body), aircraft-type dependent. |

---

## 3. Data Requirements

### 3.1 Inputs — ADS-B (per flight)
| Field | Notes |
|-------|-------|
| Position latitude / longitude | From position message; has its **own timestamp**, distinct from velocity. |
| Groundspeed | From velocity message; primary horizontal-kinematics signal. |
| Track | Heading over ground; used for runway alignment confirmation. |
| Barometric altitude | Present; QNH-sensitive near surface. |
| Geometric altitude | Present; preferred for absolute height near ground. |
| Barometric vertical rate | **Sometimes null**; treat as corroborating, not primary. |
| On-ground flag | Delayed transition; usable only as a coarse bracket. |
| Timestamps | **Position and velocity carry different timestamps** (source 1). Must be preserved, not silently merged. |

### 3.2 Inputs — Runway / Aircraft
| Field | Notes |
|-------|-------|
| Runway length | Used for sanity bounds and overrun context. |
| Threshold lat/long | Reference for along-runway distance. |
| Runway heading / centerline | Required for projection (derive from threshold + opposite end or survey data). |
| Runway / threshold elevation | Required for the vertical-crossing method; if not supplied, must be sourced. |
| Aircraft type | Drives approach speed priors, braking profile, and antenna lever-arm correction. |

### 3.3 Truth Data
- 40,500 QAR-matched flights with a touchdown reference (time and/or position).
- Each QAR record must be **join-keyable** to its ADS-B flight without ambiguity.
- QAR and ADS-B clocks must be reconcilable to a common time reference (or an estimable offset).

### 3.4 Data Quality Expectations
- The system must tolerate **missing barometric vertical rate**.
- The system must tolerate **irregular and asynchronous** sampling (4–5 s nominal, with gaps).
- The system must flag and handle **dropouts, duplicate samples, and outliers** in position/velocity.
- The system must not assume position and velocity are co-timed.

---

## 4. Functional Requirements

- **FR-1 — Touchdown time.** For each landing the system shall output an estimated `t_td` with sub-sample resolution (finer than the 4–5 s cadence).
- **FR-2 — Touchdown position.** The system shall output along-runway distance from threshold (m) and lateral offset from centerline (m) at `t_td`.
- **FR-3 — Uncertainty.** Each estimate shall carry a quantified uncertainty (e.g., a confidence interval or predictive distribution).
- **FR-4 — Multiple methods.** The system shall implement at least one physics/state-estimation estimator, one change-point estimator, and one learned estimator, plus a fusion/ensemble.
- **FR-5 — Interpretable anchor.** At least one estimator shall be physically interpretable and usable as a fallback for rare aircraft types or the second ADS-B source.
- **FR-6 — Aircraft-type awareness.** Estimators shall incorporate aircraft type (priors, lever-arm correction, and/or model conditioning).
- **FR-7 — Source awareness.** The system shall support both ADS-B sources and report performance per source.
- **FR-8 — Validation harness.** The system shall produce stratified error reports against QAR truth (see §5).
- **FR-9 — Reproducibility.** Given the same inputs, configuration, and random seeds, the system shall reproduce identical outputs.
- **FR-10 — Failure signaling.** When inputs are insufficient (e.g., excessive dropout near touchdown), the system shall emit a low-confidence / no-estimate flag rather than a silent guess.

---

## 5. Accuracy and Validation Requirements

### 5.1 Metrics
- Primary: **along-runway distance error** (signed, in meters) — signed so long/short bias is visible.
- Secondary: **touchdown time error** (seconds) — cleaner diagnostic of estimator quality.
- Report **full error distributions**, not only mean/RMSE: median, IQR, and **95th/99th-percentile long-side error** (overrun risk lives in the tail).

### 5.2 Targets (to be ratified with stakeholders)
- These are placeholder targets pending baseline results; they must be confirmed, not assumed.
  - Median absolute along-runway error: target ≤ 75 m.
  - 95th-percentile absolute along-runway error: target ≤ 200 m.
  - No systematic long/short bias greater than the cadence-limited floor for the dominant aircraft types.
- A naive "first on-ground sample" baseline must be computed to demonstrate lift; the system must materially beat it.

### 5.3 Validation Protocol
- **Grouped splits.** Train/validation/test splits shall be grouped by aircraft tail, airport, and runway to prevent context leakage. Random splitting is prohibited.
- **Cross-source validation.** Models trained on one ADS-B source shall be evaluated on the other to quantify domain shift.
- **Stratified reporting.** Errors shall be broken down by aircraft type, ADS-B source, airport/runway, and approach speed band.
- **Tail focus.** Reporting shall emphasize the long-landing tail relevant to overrun safety.
- **Truth-quality audit.** A sample of QAR-to-ADS-B time alignment shall be audited to bound label noise.

---

## 6. Non-Functional Requirements

- **NFR-1 — Explainability.** Because outputs feed safety analysis, every reported estimate shall be traceable to either a physical model or, for learned models, accompanied by an interpretable anchor estimate and feature attributions. Unexplained black-box numbers are not acceptable as the sole output.
- **NFR-2 — Scalability.** The pipeline shall process the full 40,500-flight corpus (and larger) in batch within practical compute budgets.
- **NFR-3 — Configurability.** Method selection, smoothing windows, lever-arm tables, and thresholds shall be configuration-driven, not hard-coded.
- **NFR-4 — Versioning.** Data version, code version, model artifact, and config shall be recorded with every output set for auditability.
- **NFR-5 — Testability.** Core kinematic transforms (projection, interpolation, lever-arm correction) shall have unit tests with known-answer cases.

---

## 7. Assumptions

- QAR provides a reliable, low-noise touchdown reference whose clock can be reconciled to ADS-B.
- Runway threshold coordinates and runway elevation are accurate and current (no recent displaced-threshold or survey changes unaccounted for).
- Aircraft type is correctly identified per flight and an antenna lever-arm value is available or estimable per type.
- The 40,500 flights are representative of the operational population of interest (subject to the skew audit in Risks).
- ADS-B geometric altitude is available for the great majority of landings.

---

## 8. Risks (requirements-level)

- **R-1 — Label/clock misalignment.** A constant or drifting offset between QAR and ADS-B clocks would bias all learned and calibrated estimates. Mitigation: explicit offset estimation and audit (see Design §error-sources).
- **R-2 — Type/source skew.** 40,500 flights may be dominated by a few types/airports/sources; rare strata may be under-served. Mitigation: distribution audit; physics fallback for thin strata.
- **R-3 — Leakage via random splits.** Inflated, non-generalizing metrics. Mitigation: mandatory grouped splits (FR/§5.3).
- **R-4 — Over-reliance on a single signal.** Null vertical rate or geometric-altitude bias can sink any single-signal method. Mitigation: multi-method ensemble (FR-4).
- **R-5 — Cadence-limited floor.** At 4–5 s, sub-sample estimation has an irreducible variance floor; targets must respect it. Mitigation: characterize the floor empirically and set targets accordingly.
- **R-6 — Displaced thresholds / wrong reference.** Using the geometric runway start instead of the displaced landing threshold biases distance. Mitigation: validate threshold definitions per runway.

---

## 9. Open Questions

- Is the reported touchdown the **first** main-gear contact, or should bounces produce multiple events?
- What is the exact QAR truth definition (WoW vs radio-alt = 0) and its timestamp precision?
- Are antenna lever-arm values available per type, or must they be calibrated from data?
- What are the ratified accuracy acceptance thresholds (§5.2)?
- Is an online/real-time variant required later (affects architecture choices now)?
