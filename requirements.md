# Requirements Document

> **Revision v1.1 — review changes incorporated.** Added: SI units convention; HAE/MSL vertical-datum convention and geoid correction (Req 11, 17); two-pass coarse-bracket and flag-independent trigger (Req 1); trajectory classification for go-around/touch-and-go/bounce (Req 21); per-source capability descriptor with FR24 barometric-only/interpolated assumption (Req 8); nominal-pitch lever-arm geometry and class-median (non-worst-case) missing-lever-arm default (Req 2, 7); extended vertical-crossing fit region (Req 17); segmented-regression-first derivative policy (Req 16); cross-correlation clock alignment with clock-independent distance truth (Req 12, 19); separated split protocol with calibration partition (Req 12); two-sided CI coverage (Req 4); wrong-runway lateral-offset gate (Req 2); provisional accuracy targets pending a cadence-floor baseline (Req 13); relaxed bit-identical reproducibility for neural components (Req 15). **Open item:** FR24 altitude/sample provenance unconfirmed — encoded as a config-gated assumption.

## Introduction

This system infers aircraft touchdown point from ADS-B surveillance data for landing safety analysis. Touchdown — the moment of first main-gear contact with the runway — is not directly observed in ADS-B. The system estimates touchdown time to sub-sample resolution, maps it to an along-runway position (distance from threshold), and quantifies uncertainty. Estimates support identification of long-landing and high-speed-touchdown operational hazards that increase runway excursion and overrun risk.

The system operates in offline batch mode over historical ADS-B trajectories, validated against 40,500 QAR-matched flights with known touchdown positions. The current dataset contains Boeing aircraft models; the architecture accommodates future expansion to other manufacturers.

### Units Convention

To avoid unit-confusion defects, the system SHALL hold all internal computation in SI units (meters, meters/second, seconds, radians) with a single conversion to presentation units (feet, knots) performed only at the output boundary. Every configuration value, lookup-table entry, and interface field SHALL carry an explicit unit suffix. Distances in this document are stated in feet where they describe analyst-facing outputs/targets and in meters where they describe internal geometry; the two are bridged only at the output layer.

### Vertical Datum Convention

ADS-B Geometric_Altitude is height above the WGS-84 ellipsoid (HAE). Published runway/airport elevations are almost always orthometric (height above mean sea level / the geoid), and the two differ by the local geoid undulation (commonly −15 to −35 m over the continental United States, up to roughly ±100 m globally). All vertical comparisons between ADS-B geometric altitude and runway elevation SHALL be performed in a single, explicitly declared datum, with geoid correction applied where the source elevation is orthometric (see Requirement 11 and Requirement 17). Mixing HAE and MSL without correction would inject a tens-of-meters vertical bias directly into the vertical-crossing estimator.

## Glossary

- **Touchdown_Detector**: The complete system that estimates touchdown time and position from ADS-B data
- **Touchdown**: First main-gear contact with the runway surface during landing
- **Touchdown_Time**: The estimated time of touchdown (`t_td`), resolved to sub-sample precision (finer than the 4–5 second ADS-B update cadence)
- **Touchdown_Point**: The horizontal runway position at touchdown time, expressed as along-runway distance from the landing threshold (feet)
- **Threshold**: The beginning of the runway surface usable for landing; the reference origin for along-runway distance, specified as a lat/long coordinate
- **ADS-B**: Automatic Dependent Surveillance–Broadcast; surveillance data providing aircraft position and velocity at 4–5 second update intervals
- **Geometric_Altitude**: GNSS-derived height above the WGS-84 ellipsoid; preferred for absolute height determination near ground
- **Barometric_Altitude**: Pressure-derived altitude; subject to QNH setting error near the surface
- **On_Ground_Flag**: ADS-B air/ground status bit; transitions with variable delay after actual touchdown
- **QAR**: Quick Access Recorder; onboard flight data recorder providing ground-truth touchdown timestamp and lat/long position
- **Update_Cadence**: The nominal ADS-B sample interval of 4–5 seconds between successive observations
- **Lever_Arm**: The physical offset between the GNSS antenna location and the main landing gear contact point, varying by aircraft type; has both a vertical component (antenna height above gear) and a longitudinal component (antenna forward/aft of gear). Resolving the lever arm into a ground-distance correction also requires the touchdown pitch attitude (see Touchdown_Pitch)
- **Touchdown_Pitch**: The aircraft pitch attitude at the moment of touchdown. Pitch is NOT observable in ADS-B; the system uses a per-aircraft-type nominal touchdown pitch constant, sourced from the lever-arm/type configuration table, to project the longitudinal lever arm onto the runway
- **Geoid_Undulation**: The height difference between the WGS-84 ellipsoid and the geoid (mean sea level) at a given location; used to convert orthometric (MSL) runway elevations to HAE for comparison against ADS-B geometric altitude
- **Aircraft_Class**: A coarse grouping of aircraft types by size/wake category (e.g., regional, narrowbody, widebody) used to select a default Lever_Arm when a type-specific value is unavailable
- **Trajectory_Type**: A classification of an input trajectory as a completed landing, a go-around (no touchdown), or a touch-and-go (brief contact without sustained ground roll). Only completed landings produce a touchdown estimate
- **Along_Runway_Distance**: Distance in feet measured along the runway centerline from the landing threshold to the touchdown point
- **Lateral_Offset**: Perpendicular distance from the runway centerline at the touchdown point
- **Physics_Estimator**: An estimator based on kinematic models (deceleration-knee, vertical crossing, state-estimation) that is physically interpretable
- **Change_Point_Estimator**: An estimator that detects regime transitions in deceleration or jerk signals (CUSUM, PELT, jerk-onset)
- **Learned_Estimator**: An estimator trained on QAR-labeled data (gradient-boosted trees, neural sequence models)
- **Fusion_Ensemble**: A calibrated combination of multiple estimator outputs into a single estimate with uncertainty
- **Grouped_Split**: A train/test partition where all flights sharing a tail number, airport, or runway appear in only one partition

## Requirements

### Requirement 1: Touchdown Time Estimation

**User Story:** As a safety analyst, I want the system to estimate when an aircraft touches down to sub-sample resolution, so that I can determine precise landing performance from coarse ADS-B data.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL operate in two passes: a first pass that establishes a coarse touchdown bracket (a bounded time window expected to contain `t_td`), and a second pass that produces the sub-sample `t_td` estimate within that bracket. All acceptance criteria that reference "the estimated touchdown region" or "the estimated touchdown time" for data-sufficiency or quality gating SHALL be evaluated against the coarse bracket from the first pass, not against the final estimate (which is not yet available when gating occurs)
2. THE Touchdown_Detector SHALL establish the coarse touchdown bracket using the On_Ground_Flag transition as an upper time-bound where available, AND a flag-independent indicator (the descent of Geometric_Altitude toward runway elevation together with the onset of ground-roll deceleration) so that a bracket can still be formed when the On_Ground_Flag is absent, delayed, or missing
3. WHEN a valid completed-landing trajectory (per Requirement 21) is provided containing at least 3 samples with groundspeed values within 60 seconds of the coarse touchdown bracket, THE Touchdown_Detector SHALL estimate a Touchdown_Time with resolution of 1 second or finer, accompanied by a quantified uncertainty interval
4. THE Touchdown_Detector SHALL define touchdown as the moment of first main-gear contact with the runway surface. WHERE a bounce produces more than one contact, the reported `t_td` SHALL correspond to the first contact, and the estimators SHALL NOT report a value averaged across a bounce (see Requirement 21)
5. WHEN the ADS-B trajectory contains fewer than 3 samples that pass kinematic-gate quality checks within 30 seconds of the coarse touchdown bracket, THE Touchdown_Detector SHALL output the estimate with a low-confidence flag and the reason for degraded confidence
6. IF the ADS-B trajectory contains no samples within 60 seconds of the coarse touchdown bracket or lacks groundspeed data entirely, THEN THE Touchdown_Detector SHALL emit a no-estimate flag instead of producing a touchdown time

### Requirement 2: Touchdown Position Output

**User Story:** As a safety analyst, I want the touchdown mapped to a distance along the runway from the threshold, so that I can identify long landings that increase overrun risk.

#### Acceptance Criteria

1. WHEN a Touchdown_Time is estimated, THE Touchdown_Detector SHALL compute the Along_Runway_Distance from the landing Threshold in feet by interpolating the horizontal trajectory at the estimated Touchdown_Time and projecting the interpolated position onto the runway centerline
2. WHEN a Touchdown_Time is estimated, THE Touchdown_Detector SHALL compute the Lateral_Offset from the runway centerline in feet as a secondary output
3. THE Touchdown_Detector SHALL apply the Lever_Arm correction per aircraft type when mapping time to position, including the vertical component (antenna height above main gear, applied to the altitude crossing) and the horizontal ground-distance correction evaluated at the nominal Touchdown_Pitch θ as (longitudinal_offset · cos θ + vertical_offset · sin θ), applied to along-runway distance. Pitch is assumed from the per-type configuration, not measured (see Requirement 7)
4. IF the computed Along_Runway_Distance exceeds the runway length or is negative, THEN THE Touchdown_Detector SHALL flag the estimate as out-of-bounds and include the computed value in the output for diagnostic purposes
5. IF the computed Lateral_Offset exceeds half the runway width plus a configurable margin (default: runway half-width + 50 ft), THEN THE Touchdown_Detector SHALL flag the estimate with a suspected-wrong-runway indicator, since a large lateral offset typically signals runway mis-assignment (e.g., a parallel-runway swap) or erroneous geometry rather than a genuine off-centerline touchdown

### Requirement 3: Touchdown Speed Output

**User Story:** As a safety analyst, I want the groundspeed at touchdown, so that I can identify high-speed touchdowns that increase stopping-distance risk.

#### Acceptance Criteria

1. WHEN a Touchdown_Time is estimated, THE Touchdown_Detector SHALL output the interpolated groundspeed at Touchdown_Time in knots, rounded to 0.1 kt resolution, within the physically plausible range of 50 to 220 knots
2. THE Touchdown_Detector SHALL derive the touchdown speed from the kinematic interpolation used for position mapping, not from a single nearest ADS-B sample
3. WHEN a touchdown speed is output, THE Touchdown_Detector SHALL additionally report a speed uncertainty interval derived from the propagation of Touchdown_Time uncertainty through the interpolated groundspeed profile
4. IF the kinematic interpolation cannot produce a groundspeed value within the plausible range of 50 to 220 knots or the surrounding ADS-B velocity samples are missing within 10 seconds of the estimated Touchdown_Time, THEN THE Touchdown_Detector SHALL flag the speed estimate as low-confidence and include a reason indicator

### Requirement 4: Uncertainty Quantification

**User Story:** As a safety analyst, I want each touchdown estimate to carry a confidence interval, so that I can assess the reliability of individual estimates and appropriately weight them in risk analysis.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL output a 90% confidence interval, in seconds, for each Touchdown_Time estimate, where the interval bounds represent the range within which the true touchdown time is expected to lie with 90% probability
2. THE Touchdown_Detector SHALL output a 90% confidence interval, in feet, for each Along_Runway_Distance estimate, where the interval bounds represent the range within which the true along-runway distance is expected to lie with 90% probability
3. WHEN validated against QAR truth data across the held-out test set, THE Touchdown_Detector 90% confidence intervals for Touchdown_Time SHALL achieve empirical coverage between 85% and 95% (bounding both undercoverage, which is unsafe, and overcoverage, which indicates uninformative intervals)
4. WHEN validated against QAR truth data across the held-out test set, THE Touchdown_Detector 90% confidence intervals for Along_Runway_Distance SHALL achieve empirical coverage between 85% and 95%. NOTE: Along_Runway_Distance truth is derived from the QAR touchdown lat/long and is therefore independent of QAR–ADS-B clock alignment; coverage of the distance interval SHALL be assessed without applying any clock-offset correction
5. IF the Touchdown_Detector determines that a reliable uncertainty interval cannot be computed for a given flight (due to excessive data dropout or insufficient observations near touchdown), THEN THE Touchdown_Detector SHALL flag that estimate as low-confidence and output the interval with the flag rather than suppressing the uncertainty information

### Requirement 5: Multiple Estimation Methods

**User Story:** As a safety analyst, I want multiple independent estimation approaches combined in an ensemble, so that the system is robust to failures of any single signal or method.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL implement at least one Physics_Estimator that estimates touchdown from kinematic signals (deceleration profile, vertical trajectory, or state-estimation mode transition)
2. THE Touchdown_Detector SHALL implement at least one Change_Point_Estimator that detects the regime transition in deceleration or jerk signals
3. THE Touchdown_Detector SHALL implement at least one Learned_Estimator trained on QAR-labeled data
4. THE Touchdown_Detector SHALL implement a Fusion_Ensemble that combines outputs from at least three estimator families (physics, change-point, and learned) into a single fused estimate with a predictive interval whose nominal coverage is validated against held-out QAR data (stated 90% intervals shall achieve between 85% and 95% empirical coverage)
5. WHEN an individual estimator reports sigma_t exceeding a configured confidence threshold or emits a failure diagnostic, THE Fusion_Ensemble SHALL reduce that estimator's weight to below its nominal weight or exclude it entirely for the affected flight, and record which estimators were down-weighted or excluded in the output diagnostics
6. IF all individual estimators produce failed or below-threshold-confidence estimates for a flight, THEN THE Fusion_Ensemble SHALL emit a no-estimate flag rather than producing an unreliable fused result

### Requirement 6: Interpretable Physical Anchor

**User Story:** As a safety analyst, I want at least one estimator to be physically interpretable, so that I can explain results to regulators and use a trustworthy fallback for rare aircraft types.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL provide at least one Physics_Estimator whose output includes the estimated touchdown time, its uncertainty, and the named kinematic quantities used in derivation (e.g., fitted deceleration segments for decel-knee, altitude crossing point for flare-crossing, or mode probabilities for IMM)
2. WHEN the Learned_Estimator is used as the primary output, THE Touchdown_Detector SHALL include the Physics_Estimator touchdown time, uncertainty, and diagnostic quantities in the same output record
3. WHILE processing an aircraft type with fewer than 50 QAR-labeled flights in the training set (counted per aircraft type regardless of ADS-B source), THE Touchdown_Detector SHALL use the Physics_Estimator as the primary output rather than the Learned_Estimator
4. IF the Physics_Estimator cannot produce a valid estimate for a flight of a rare aircraft type (fewer than 50 training flights), THEN THE Touchdown_Detector SHALL emit a low-confidence flag and omit a touchdown estimate rather than falling back to the Learned_Estimator

### Requirement 7: Aircraft Type Awareness

**User Story:** As a safety analyst, I want the system to account for aircraft-type differences, so that estimates are accurate across the fleet mix (different approach speeds, sizes, and antenna positions).

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL accept aircraft type as an ICAO type designator input for each flight
2. WHEN mapping Touchdown_Time to Touchdown_Point, THE Touchdown_Detector SHALL apply the aircraft-type-specific vertical Lever_Arm correction (antenna height above main gear) to the altitude-crossing computation and the longitudinal Lever_Arm correction (antenna forward/aft offset from main gear) to the along-runway position. The horizontal ground-distance correction SHALL include both the longitudinal term and the height-induced term at the nominal touchdown pitch θ, i.e. (longitudinal_offset · cos θ + vertical_offset · sin θ), rather than the longitudinal term alone
3. THE Touchdown_Detector SHALL maintain a configurable external lookup table per ICAO type designator containing at minimum: a vertical offset (meters), a longitudinal offset (meters), a nominal Touchdown_Pitch (degrees), and an Aircraft_Class label. Because pitch is not observable in ADS-B, the nominal Touchdown_Pitch from this table SHALL be used to resolve the longitudinal lever arm, and the estimate diagnostics SHALL record that pitch was assumed (not measured)
4. IF a type-specific Lever_Arm value is not available for a given aircraft type, THEN THE Touchdown_Detector SHALL apply the MEDIAN vertical offset, longitudinal offset, and nominal pitch of the aircraft's Aircraft_Class (falling back to the global median only if the class is unknown). This central default SHALL NOT introduce a directional (long or short) bias into the touchdown distance. The Touchdown_Detector SHALL additionally (a) mark the estimate low-confidence with a missing-lever-arm reason code, and (b) inflate the Along_Runway_Distance confidence interval to span the full plausible lever-arm range for that class
5. THE Touchdown_Detector SHALL NOT use a worst-case (largest-offset) lever-arm default, because a uniformly short-biased distance produces false negatives on long landings (the overrun hazard the system exists to detect) and a uniformly long-biased default produces false positives; a central default with honestly widened uncertainty is required instead
6. WHEN producing an estimate with a missing-lever-arm indicator, THE Touchdown_Detector SHALL include the assumed default Lever_Arm values, the Aircraft_Class used, and the assumed nominal pitch in the estimate diagnostics

### Requirement 8: ADS-B Source Handling

**User Story:** As a safety analyst, I want the system to handle ADS-B data from multiple providers (Aireon, FlightRadar24), so that I can process all available surveillance data consistently.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL accept ADS-B data where position and velocity messages carry separate timestamps (asynchronous messages) and produce a touchdown estimate using the same output contract (touchdown time, uncertainty, and diagnostics) as for co-timed sources
2. THE Touchdown_Detector SHALL accept ADS-B data where position and velocity are co-timed in a single record and produce a touchdown estimate using the same output contract (touchdown time, uncertainty, and diagnostics) as for asynchronous sources
3. WHILE processing asynchronous ADS-B sources, THE Touchdown_Detector SHALL preserve the distinct position and velocity timestamps throughout the pipeline and use kinematic interpolation rather than treating the nearest position and velocity pair as simultaneous
4. THE Touchdown_Detector SHALL record which ADS-B source (Aireon or FlightRadar24) was used for each touchdown estimate in the output record
5. IF a source-specific field required by an estimator is unavailable for a given ADS-B source, THEN THE Touchdown_Detector SHALL exclude that estimator from the fusion for the affected flight and note the exclusion in the diagnostics output
6. THE Touchdown_Detector SHALL maintain a per-source capability descriptor declaring, at minimum: whether the source provides true Geometric_Altitude (HAE) versus barometric altitude only, and whether position samples are raw observations versus provider-interpolated/smoothed. Estimators SHALL be enabled or disabled per flight according to this descriptor
7. ASSUMPTION (to be confirmed): the FlightRadar24 source provides barometric altitude only (no true geometric altitude) and provider-interpolated rather than raw samples. WHILE this assumption holds, THE Touchdown_Detector SHALL disable all estimators that depend on geometric altitude (e.g., vertical flare-crossing, geometric IMM altitude updates) for FlightRadar24 flights, SHALL NOT treat FlightRadar24 samples as independent raw observations in any noise/independence assumption, and SHALL flag this source's estimates to reflect the reduced estimator set. This behavior SHALL be driven entirely by the capability descriptor (criterion 6) so it can be reconfigured once the source's true characteristics are confirmed
8. WHEN a source provides only barometric altitude, THE Touchdown_Detector SHALL NOT substitute barometric altitude into the geometric-altitude vertical-crossing computation as if it were geometric (the two differ near the surface and have different datums)

### Requirement 9: Data Quality Tolerance

**User Story:** As a safety analyst, I want the system to handle real-world ADS-B data quality issues gracefully, so that it produces results where possible and fails transparently where it cannot.

#### Acceptance Criteria

1. WHEN barometric vertical rate is missing or null, THE Touchdown_Detector SHALL still produce an estimate using remaining signals (groundspeed, geometric altitude, position) and SHALL indicate in the output diagnostics which signals were unavailable
2. WHEN the ADS-B trajectory contains gaps exceeding 10 seconds within ±30 seconds of the estimated touchdown time, THE Touchdown_Detector SHALL widen the reported confidence interval by at least a factor proportional to the ratio of gap duration to nominal sample interval (e.g., a 10-second gap with 5-second nominal cadence doubles the interval width)
3. WHEN two or more ADS-B samples share identical timestamps (within 0.1 seconds), THE Touchdown_Detector SHALL retain only one sample per timestamp (selecting the last-received) before processing
4. WHEN position or velocity values imply longitudinal acceleration exceeding 1.0 g, lateral acceleration exceeding 0.5 g, or turn rate exceeding 6 degrees per second, THE Touchdown_Detector SHALL exclude the offending samples and record the count and timestamps of excluded samples in the output diagnostics
5. IF ADS-B data quality is insufficient to produce an estimate (fewer than 3 valid samples within ±30 seconds of the estimated touchdown time, or a continuous gap exceeding 15 seconds that spans the estimated touchdown time, or more than 50 percent of samples within that window excluded by plausibility checks), THEN THE Touchdown_Detector SHALL emit a no-estimate flag with a reason code identifying which condition triggered the rejection
6. WHEN the on-ground flag transitions but fewer than 2 valid position samples exist after the transition within 15 seconds, THE Touchdown_Detector SHALL increase the reported uncertainty to reflect the inability to confirm ground-roll kinematics

### Requirement 10: Asynchronous Timestamp Handling

**User Story:** As a data engineer, I want the system to correctly handle the fact that ADS-B position and velocity have different emission times, so that naive merging does not inject position errors comparable to the touchdown distance itself.

#### Acceptance Criteria

1. IF position and velocity messages originate from asynchronous ADS-B sources, THEN THE Touchdown_Detector SHALL store position timestamps and velocity timestamps as separate fields without merging them into a single sample time
2. WHEN aligning position and velocity to a common query time for downstream estimation, THE Touchdown_Detector SHALL propagate position using velocity (dead-reckoning interpolation) rather than selecting the nearest sample by time
3. THE Touchdown_Detector SHALL not introduce position errors exceeding 30 feet (9.14 meters) due to timestamp misalignment, verified by unit tests using synthetic trajectories with known timestamp offsets at groundspeeds between 120 and 150 knots
4. IF velocity data is unavailable or null at the time required for kinematic interpolation, THEN THE Touchdown_Detector SHALL flag the affected sample as degraded and fall back to linear positional interpolation between the two nearest valid position messages

### Requirement 11: Runway Reference Data

**User Story:** As a data engineer, I want the system to use precise runway geometry, so that along-runway distances are measured from the correct landing threshold and projected along the correct heading.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL accept runway threshold coordinates (latitude and longitude in decimal degrees with at least 6 decimal places), runway heading (degrees true north, 0.00–359.99), threshold elevation, runway length (meters), and runway width (meters) as inputs for each landing. The threshold elevation input SHALL be accompanied by an explicit datum tag (HAE or MSL/orthometric)
2. IF the threshold elevation is provided in an orthometric (MSL) datum, THEN THE Touchdown_Detector SHALL convert it to HAE by adding the local Geoid_Undulation (from a configured geoid model, e.g. EGM2008) before any comparison against ADS-B Geometric_Altitude. THE Touchdown_Detector SHALL NOT compare an orthometric runway elevation directly against HAE geometric altitude
3. THE Touchdown_Detector SHALL measure Along_Runway_Distance from the landing threshold (displaced threshold where one is defined for the runway), not from the physical runway start, using a sign convention where positive values indicate distance past the threshold in the landing direction and negative values indicate touchdown prior to the threshold
4. THE Touchdown_Detector SHALL use geodesic or local tangent plane (ENU) coordinate math for projection, not naive Euclidean distance on raw latitude/longitude values, such that projection error does not exceed 0.1 m over a distance equal to the runway length
5. IF any of the required runway reference fields (threshold latitude, threshold longitude, runway heading, threshold elevation, runway length, or runway width) is missing, null, or outside valid bounds (latitude ±90, longitude ±180, heading 0–360, elevation −500 to 10000 m, length 0 to 6000 m, width 0 to 100 m) for a given flight, THEN THE Touchdown_Detector SHALL reject that flight with an error indication identifying which field is missing or invalid, and SHALL NOT produce a touchdown estimate for that flight
6. WHEN a runway has a displaced threshold, THE Touchdown_Detector SHALL use the displaced threshold coordinates as the reference origin, not the physical runway start coordinates, and the Along_Runway_Distance SHALL reflect the distance from that displaced threshold

### Requirement 12: Validation Against QAR Truth

**User Story:** As a safety analyst, I want rigorous validation against QAR ground truth, so that I can trust the system's accuracy claims and identify where it performs poorly.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL provide a validation harness that computes error metrics against QAR truth data (touchdown timestamp and lat/long position), reporting at minimum: signed distance error, absolute distance error, time error, RMSE, median absolute error, interquartile range, 95th-percentile absolute error, and 99th-percentile absolute error
2. THE Touchdown_Detector SHALL use a primary split grouped by aircraft tail number for the headline accuracy metrics — no flight in the test set shall share a tail number with any flight in the training set. Tail-grouping is the leakage control that prevents the model from memorizing an individual airframe's sensor biases
3. THE Touchdown_Detector SHALL additionally report two generalization-stress evaluations: a held-out-airport evaluation (test airports absent from training) and a held-out-runway evaluation (test runways absent from training), reported alongside the primary tail-grouped metrics rather than intersected into a single split. Rationale: intersecting tail, airport, and runway grouping into one partition needlessly starves the usable data and conflates leakage control with geographic generalization, which are distinct questions
4. THE Touchdown_Detector SHALL use a three-way partition (train / calibration / test) where the calibration partition, disjoint from both train and test under the same grouping rule, is used to fit/conformalize the uncertainty intervals, so that reported coverage is measured on data unseen during both model fitting and interval calibration
5. THE Touchdown_Detector SHALL report signed Along_Runway_Distance error so that systematic long or short bias is visible, with positive error indicating the estimate is longer (farther from threshold) than truth
6. THE Touchdown_Detector SHALL report the full error distribution including median, interquartile range, 95th-percentile, and 99th-percentile absolute error, plus the 95th-percentile positive (long-side) signed error
7. THE Touchdown_Detector SHALL report error metrics stratified by aircraft type, ADS-B source, airport, and approach speed band, where each stratum contains at least 30 flights to be reportable
8. THE Touchdown_Detector SHALL compute a naive baseline (first on-ground sample position) and demonstrate that the system materially outperforms it, reporting both baseline and system metrics side-by-side
9. THE Touchdown_Detector SHALL evaluate cross-source generalization by training on one ADS-B source and testing on the other, reporting the same metrics as the primary validation
10. THE Touchdown_Detector SHALL compute Along_Runway_Distance truth directly from the QAR touchdown lat/long (a clock-independent geometric quantity) and SHALL report distance metrics without relying on QAR–ADS-B clock alignment; clock alignment (Requirement 19) is required only for time-domain labels and the time-error metric

### Requirement 13: Accuracy Targets

**User Story:** As a safety analyst, I want the system to meet defined accuracy thresholds, so that estimates are precise enough for operational hazard identification.

> **STATUS — PROVISIONAL TARGETS.** The numeric thresholds in this requirement are provisional and SHALL be ratified after a baseline run characterizes the irreducible, cadence-limited error floor (the variance attributable solely to the 4–5 s update interval at approach groundspeeds). Until ratified, these values define reporting targets, not pass/fail compliance gates; the system SHALL report observed metrics against them rather than be deemed non-compliant. As a reference point, a 250 ft RMSE corresponds to roughly a 1 s time RMSE at typical approach groundspeed, which is aggressive relative to the cadence and must be empirically confirmed achievable.

#### Acceptance Criteria

0. THE Touchdown_Detector SHALL empirically characterize and report the cadence-limited error floor before the targets below are ratified, and SHALL set/confirm targets to respect that floor
1. THE Touchdown_Detector SHALL report (target: at or below 250 feet, provisional) the overall RMSE of Along_Runway_Distance error across all flights in the held-out test set when validated against QAR truth, where the test set contains a minimum of 4,000 flights and is split according to the grouped-split protocol defined in the validation requirements
2. THE Touchdown_Detector SHALL not exhibit a systematic long or short bias exceeding 75 feet in median signed error for any aircraft type representing more than 5% of the test set
3. THE Touchdown_Detector SHALL report the 95th-percentile absolute Along_Runway_Distance error, with a target at or below 400 feet, and the 95th-percentile positive (long-side) signed Along_Runway_Distance error shall not exceed 500 feet
4. WHEN the naive baseline (first on-ground sample) is computed on the same test set, THE Touchdown_Detector SHALL demonstrate at least 30% reduction in RMSE compared to the baseline
5. IF the Touchdown_Detector fails to meet any target defined in criteria 1 through 3 for a specific ADS-B source or aircraft-type stratum containing at least 200 flights, THEN THE Touchdown_Detector SHALL flag that stratum as below-target in the validation report along with the observed metric values

### Requirement 14: Failure Signaling

**User Story:** As a safety analyst, I want the system to clearly signal when it cannot produce a reliable estimate, so that I do not unknowingly use unreliable data in risk assessments.

#### Acceptance Criteria

1. WHEN input data is insufficient for a credible estimate (as defined by the data quality thresholds in Requirement 9), THE Touchdown_Detector SHALL emit a no-estimate flag with a machine-readable reason code from a defined enumeration of failure reasons
2. WHEN the estimated 90% confidence interval for Along_Runway_Distance exceeds a configurable width threshold (default: 600 feet), THE Touchdown_Detector SHALL flag the estimate as low-confidence
3. THE Touchdown_Detector SHALL never produce a touchdown estimate without an accompanying confidence classification of either "normal", "low-confidence", or "no-estimate"
4. THE Touchdown_Detector SHALL include in each output record: the confidence classification, the reason code (if low-confidence or no-estimate), and the list of estimators that contributed to the fusion (or failed)

### Requirement 15: Reproducibility

**User Story:** As a data scientist, I want identical outputs given identical inputs, so that results are auditable and debugging is possible.

#### Acceptance Criteria

1. WHEN given the same input data, configuration, random seeds, and software environment (Python version, library versions), THE Touchdown_Detector SHALL produce bit-identical outputs for all numeric fields originating from the physics, change-point, LightGBM, and geometry components. For neural sequence-model components, where exact bit-reproducibility is generally infeasible on GPU due to nondeterministic parallel reductions, THE Touchdown_Detector SHALL EITHER (a) reproduce outputs within a documented numerical tolerance, OR (b) reproduce them bit-identically when an explicit deterministic-execution mode is enabled (accepting reduced throughput). The chosen mode SHALL be recorded with the output
2. THE Touchdown_Detector SHALL propagate a single master random seed to all stochastic components (model initialization, data shuffling, dropout) such that setting one seed value determines all random behavior
3. THE Touchdown_Detector SHALL record with every output batch: data version identifier, git commit hash of the code, model artifact hash, resolved configuration (including defaults), Python version, and versions of key numerical libraries (NumPy, SciPy, scikit-learn, PyTorch/TensorFlow if used, LightGBM)

### Requirement 16: Derivative Signal Quality

**User Story:** As a data scientist, I want velocity derivatives (deceleration, jerk) computed with appropriate smoothing, so that the 4–5 second cadence does not produce noise-dominated signals.

#### Acceptance Criteria

1. THE primary deceleration-regime estimate SHALL be obtained by fitting a segmented (piecewise) model directly to the raw groundspeed-vs-time series and taking the breakpoint as the regime transition, rather than by differentiating then detecting. Rationale: differentiation-then-detection forces a smoothing-window trade-off (criterion 5) that blurs the very transition being located; fitting segments to the raw signal localizes the breakpoint without that trade-off
2. THE Touchdown_Detector SHALL compute first-derivative (deceleration) and second-derivative (jerk) of groundspeed — where derivatives are needed for corroborating signals — using either a Savitzky-Golay filter with polynomial order no greater than 3 and a window spanning at least 5 samples, or a Gaussian process, rather than naive finite differencing
3. THE Touchdown_Detector SHALL not use raw second-order finite differences (jerk) as a sole basis for determining `t_td`; jerk derived from unsmoothed finite differencing at 4–5 second cadence SHALL only be used as a corroborating signal alongside at least one other estimator
4. WHERE a Gaussian process is used for derivative estimation, THE Touchdown_Detector SHALL include the derivative posterior standard deviation in the estimator output alongside each derivative value, and downstream estimators SHALL weight the derivative inversely to its uncertainty. A single stationary length scale SHALL NOT be assumed across the whole landing; because the signal is smooth on approach and sharply non-stationary at the deceleration knee, the smoothing/GP configuration SHALL either use a non-stationary/locally-adaptive treatment or be applied piecewise so the transition is not over-smoothed
5. THE Touchdown_Detector SHALL reconcile the competing needs of derivative noise suppression (favoring a wider window) and transition locality (favoring a narrower window). The configured smoothing window SHALL be reported in diagnostics, and any estimator whose required window would smear the coarse touchdown bracket SHALL prefer the segmented-regression approach of criterion 1 over differentiate-then-detect
6. IF the number of valid groundspeed samples within the smoothing window is fewer than 5, THEN THE Touchdown_Detector SHALL flag the derivative as unreliable and emit a low-confidence indicator for any estimator that depends on that derivative
7. WHEN derivative quality is validated, THE Touchdown_Detector SHALL compare smoothed deceleration profiles against QAR-derived acceleration on a held-out sample and report the RMS discrepancy in m/s²

### Requirement 17: Vertical Profile Modeling

**User Story:** As a data scientist, I want the altitude-based estimator to model the flare correctly, so that the vertical crossing estimate is not biased long by fitting a straight line through a curved profile.

#### Acceptance Criteria

1. WHEN estimating touchdown via vertical profile crossing, THE Touchdown_Detector SHALL fit a joint glideslope-plus-flare model over an extended fit region (default: from approximately 200–300 feet above runway elevation down to the surface) and solve for the crossing, modeling the flare as a curved profile (exponential, quadratic, or piecewise-linear) rather than a single straight-line fit. Rationale: at 4–5 s cadence and typical descent rates, only one or two geometric-altitude samples normally fall below 50 ft, so a flare model constrained to the sub-50-ft region is almost always under-determined; the extended region provides 3–5 samples while the curved flare term still prevents the long bias from straight-line fitting through the flare flattening
2. THE Touchdown_Detector SHALL use Geometric_Altitude (not Barometric_Altitude) for the vertical crossing calculation, in the HAE datum, with the runway elevation converted to HAE per Requirement 11. THE vertical-crossing estimator SHALL be disabled for any source that does not provide true geometric altitude (per the Requirement 8 capability descriptor)
3. THE Touchdown_Detector SHALL treat geoid/datum correction (Requirement 11) and residual sensor bias as two separate steps: first apply the deterministic geoid conversion, THEN, only if the geoid-corrected median Geometric_Altitude of high-approach samples (well above the flare, where true height is well modeled by the glideslope) still deviates from expected by more than 15 feet, estimate and subtract a residual static sensor bias. The static-bias estimate SHALL NOT be derived from samples inside the flare region (where the profile is curving) to avoid absorbing real flare dynamics into the bias term
4. THE Touchdown_Detector SHALL subtract the aircraft-type-specific antenna-to-main-gear height from the Geometric_Altitude (or equivalently add it to the crossing target elevation) so that the crossing solution corresponds to main-gear contact rather than antenna height
5. IF fewer than 3 Geometric_Altitude samples exist within the extended fit region (criterion 1), THEN THE Touchdown_Detector SHALL flag the vertical-profile estimate as low-confidence and defer to other estimators rather than fitting an under-constrained curve

### Requirement 18: On-Ground Flag Usage

**User Story:** As a data scientist, I want the system to use the on-ground flag only as a coarse bracket, so that its known delayed transition does not bias estimates long.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL not output the On_Ground_Flag transition time as the touchdown estimate `t_td` for any landing
2. THE Touchdown_Detector SHALL treat the On_Ground_Flag transition time as an upper time-bound, constraining `t_td` such that `t_td` is less than or equal to the On_Ground_Flag transition time
3. IF any estimator produces a candidate `t_td` that exceeds the On_Ground_Flag transition time, THEN THE Touchdown_Detector SHALL discard or clamp that candidate to no later than the On_Ground_Flag transition time
4. THE Touchdown_Detector SHALL not assign the On_Ground_Flag transition time a weight greater than zero in any ensemble or fusion calculation of `t_td`

### Requirement 19: QAR Clock Alignment

**User Story:** As a data scientist, I want the system to explicitly handle clock offsets between QAR truth and ADS-B, so that a systematic time bias does not corrupt all trained and calibrated estimates.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL estimate the QAR–ADS-B clock offset for each matched flight by cross-correlating an overlapping kinematic time series common to both streams (e.g., groundspeed and/or along-track position over the approach and rollout), selecting the lag that maximizes alignment. THE Touchdown_Detector SHALL NOT align on touchdown itself, as that would be circular with the quantity being estimated
2. THE clock-offset estimation SHALL detect and report any within-flight clock drift (a lag that varies across the trajectory), not assume a single constant offset, and SHALL flag flights exhibiting drift beyond a configurable bound
3. Clock alignment SHALL be required ONLY for time-domain training labels and the time-error metric. Along_Runway_Distance truth, being derived geometrically from QAR touchdown lat/long, SHALL be computed and validated WITHOUT clock alignment (see Requirement 12). This confines clock-offset risk to the time domain and removes it from the primary distance metric
4. THE Touchdown_Detector SHALL apply the estimated clock offset correction to QAR timestamps before using them as time-domain training labels or time-error truth
5. THE Touchdown_Detector SHALL report the distribution of estimated clock offsets as a data quality diagnostic, including at minimum the median, standard deviation, and 95th-percentile absolute offset across the flight corpus, and SHALL set the per-flight exclusion threshold (criterion 6) with reference to this observed distribution rather than to an assumed value
6. IF the estimated clock offset exceeds a configurable threshold (default: 2 seconds, to be confirmed against the observed offset distribution) for a given flight, THEN THE Touchdown_Detector SHALL exclude that flight from time-domain model training and time-error validation (but MAY retain it for distance-domain validation, which is clock-independent) and record it in a flagged-flights report
7. IF the clock offset cannot be reliably estimated for a given flight due to insufficient common-series overlap or quality, THEN THE Touchdown_Detector SHALL exclude that flight from time-domain training and time-error validation and log the reason for estimation failure

### Requirement 20: Configuration-Driven Operation

**User Story:** As a data scientist, I want all tunable parameters externalized in configuration, so that experiments are reproducible and the system is adaptable without code changes.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL externalize the following parameters in configuration files: estimator selection and fusion weights, smoothing window sizes and polynomial orders, Gaussian process hyperparameters, Lever_Arm lookup values per aircraft type (including vertical offset, longitudinal offset, nominal Touchdown_Pitch, and Aircraft_Class), Aircraft_Class median lever-arm defaults, per-source capability descriptors (geometric-altitude availability, raw-vs-interpolated samples), geoid model selection, the vertical-crossing fit-region bounds, the wrong-runway lateral-offset margin, plausibility thresholds, confidence thresholds, quality-gate limits (maximum gap size, minimum sample count), clock-offset and drift thresholds, and validation split grouping definitions
2. THE Touchdown_Detector SHALL treat any numeric literal, threshold, window size, or lookup value that affects estimation output as a tunable parameter and SHALL NOT embed such values in source code
3. WHEN a configuration parameter is missing, THE Touchdown_Detector SHALL use a default value defined in a defaults section of the configuration schema and SHALL log a warning message identifying the parameter name and the default value applied
4. IF a configuration parameter value fails schema validation (wrong type, out of declared min/max range, or referencing an unknown estimator name), THEN THE Touchdown_Detector SHALL reject the configuration at startup, report an error message identifying the invalid parameter and the constraint violated, and SHALL NOT proceed with processing
5. WHEN a processing run completes, THE Touchdown_Detector SHALL record the complete resolved configuration (including any defaults applied) alongside the output so that the run can be reproduced with the identical parameter set

### Requirement 21: Trajectory Classification (Landing vs Go-Around vs Touch-and-Go)

**User Story:** As a safety analyst, I want the system to confirm that a landing actually occurred and to handle go-arounds, touch-and-goes, and bounces correctly, so that I do not get spurious touchdown estimates for trajectories that contain no (or no single) touchdown.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL classify each input trajectory as one of: completed-landing, go-around (approach with no touchdown), or touch-and-go (brief contact followed by climb-out without sustained ground roll), and SHALL record the Trajectory_Type in the output
2. IF a trajectory is classified as go-around, THEN THE Touchdown_Detector SHALL emit a no-touchdown result with a corresponding reason code rather than forcing a touchdown estimate
3. WHEN a trajectory is classified as touch-and-go, THE Touchdown_Detector SHALL either report the contact with an explicit touch-and-go indicator or emit a no-touchdown result per configuration, and SHALL NOT report it as a normal completed landing
4. WHEN a trajectory contains a bounce (more than one main-gear contact during a single landing), THE Touchdown_Detector SHALL report `t_td` at the FIRST contact and SHALL NOT report a value averaged across the bounce; estimators prone to averaging (e.g., a single smoothed crossing or a two-segment fit spanning the bounce) SHALL be constrained or flagged so they do not return a physically meaningless midpoint
5. THE Touchdown_Detector SHALL NOT assume exactly one landing per input trajectory; where the input may contain multiple landings (e.g., training circuits), the system SHALL document this as an upstream segmentation assumption or detect and handle multiple landing events explicitly
6. THE classification SHALL be validated against the QAR-derived ground truth on the matched corpus, reporting confusion between completed-landing, go-around, and touch-and-go
