# Requirements Document

## Introduction

This system infers aircraft touchdown point from ADS-B surveillance data for landing safety analysis. Touchdown — the moment of first main-gear contact with the runway — is not directly observed in ADS-B. The system estimates touchdown time to sub-sample resolution, maps it to an along-runway position (distance from threshold), and quantifies uncertainty. Estimates support identification of long-landing and high-speed-touchdown operational hazards that increase runway excursion and overrun risk.

The system operates in offline batch mode over historical ADS-B trajectories, validated against 40,500 QAR-matched flights with known touchdown positions. The current dataset contains Boeing aircraft models; the architecture accommodates future expansion to other manufacturers.

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
- **Lever_Arm**: The physical offset between the GNSS antenna location and the main landing gear contact point, varying by aircraft type; has both a vertical component (antenna height above gear) and a longitudinal component (antenna forward/aft of gear)
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

1. WHEN a valid ADS-B landing trajectory is provided containing at least 3 samples with groundspeed values within 60 seconds prior to the on-ground flag transition, THE Touchdown_Detector SHALL estimate a Touchdown_Time with resolution of 1 second or finer, accompanied by a quantified uncertainty interval
2. THE Touchdown_Detector SHALL define touchdown as the moment of first main-gear contact with the runway surface
3. WHEN the ADS-B trajectory contains fewer than 3 samples that pass kinematic-gate quality checks within 30 seconds of the estimated touchdown region, THE Touchdown_Detector SHALL output the estimate with a low-confidence flag and the reason for degraded confidence
4. IF the ADS-B trajectory contains no samples within 60 seconds of the expected touchdown region or lacks groundspeed data entirely, THEN THE Touchdown_Detector SHALL emit a no-estimate flag instead of producing a touchdown time

### Requirement 2: Touchdown Position Output

**User Story:** As a safety analyst, I want the touchdown mapped to a distance along the runway from the threshold, so that I can identify long landings that increase overrun risk.

#### Acceptance Criteria

1. WHEN a Touchdown_Time is estimated, THE Touchdown_Detector SHALL compute the Along_Runway_Distance from the landing Threshold in feet by interpolating the horizontal trajectory at the estimated Touchdown_Time and projecting the interpolated position onto the runway centerline
2. WHEN a Touchdown_Time is estimated, THE Touchdown_Detector SHALL compute the Lateral_Offset from the runway centerline in feet as a secondary output
3. THE Touchdown_Detector SHALL apply the Lever_Arm correction per aircraft type when mapping time to position, including both the vertical component (antenna height above main gear, applied to the altitude crossing) and the longitudinal component (antenna forward/aft of main gear, projected by pitch angle at touchdown, applied to along-runway distance)
4. IF the computed Along_Runway_Distance exceeds the runway length or is negative, THEN THE Touchdown_Detector SHALL flag the estimate as out-of-bounds and include the computed value in the output for diagnostic purposes

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
3. WHEN validated against QAR truth data across the full 40,500-flight corpus, THE Touchdown_Detector 90% confidence intervals for Touchdown_Time SHALL contain the QAR-derived true value in at least 85% of cases (no more than 5 percentage points of undercoverage)
4. WHEN validated against QAR truth data across the full 40,500-flight corpus, THE Touchdown_Detector 90% confidence intervals for Along_Runway_Distance SHALL contain the QAR-derived true value in at least 85% of cases (no more than 5 percentage points of undercoverage)
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
2. WHEN mapping Touchdown_Time to Touchdown_Point, THE Touchdown_Detector SHALL apply the aircraft-type-specific vertical Lever_Arm correction (antenna height above main gear) to the altitude-crossing computation and the longitudinal Lever_Arm correction (antenna forward/aft offset from main gear, projected by pitch angle at touchdown) to the along-runway position
3. THE Touchdown_Detector SHALL maintain a configurable external lookup table of Lever_Arm values per ICAO type designator, containing at minimum a vertical offset (meters) and a longitudinal offset (meters) for each entry
4. IF a Lever_Arm value is not available for a given aircraft type, THEN THE Touchdown_Detector SHALL apply the largest vertical offset and largest longitudinal offset found in the lookup table as the default (biasing the estimate toward a shorter landing distance) and include a missing-lever-arm indicator in the estimate output
5. WHEN producing an estimate for a flight with a missing-lever-arm indicator, THE Touchdown_Detector SHALL include the assumed default Lever_Arm values used in the estimate diagnostics

### Requirement 8: ADS-B Source Handling

**User Story:** As a safety analyst, I want the system to handle ADS-B data from multiple providers (Aireon, FlightRadar24), so that I can process all available surveillance data consistently.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL accept ADS-B data where position and velocity messages carry separate timestamps (asynchronous messages) and produce a touchdown estimate using the same output contract (touchdown time, uncertainty, and diagnostics) as for co-timed sources
2. THE Touchdown_Detector SHALL accept ADS-B data where position and velocity are co-timed in a single record and produce a touchdown estimate using the same output contract (touchdown time, uncertainty, and diagnostics) as for asynchronous sources
3. WHILE processing asynchronous ADS-B sources, THE Touchdown_Detector SHALL preserve the distinct position and velocity timestamps throughout the pipeline and use kinematic interpolation rather than treating the nearest position and velocity pair as simultaneous
4. THE Touchdown_Detector SHALL record which ADS-B source (Aireon or FlightRadar24) was used for each touchdown estimate in the output record
5. IF a source-specific field required by an estimator is unavailable for a given ADS-B source, THEN THE Touchdown_Detector SHALL exclude that estimator from the fusion for the affected flight and note the exclusion in the diagnostics output

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

1. THE Touchdown_Detector SHALL accept runway threshold coordinates (latitude and longitude in decimal degrees with at least 6 decimal places), runway heading (degrees true north, 0.00–359.99), threshold elevation (meters above WGS-84 ellipsoid), and runway length (meters) as inputs for each landing
2. THE Touchdown_Detector SHALL measure Along_Runway_Distance from the landing threshold (displaced threshold where one is defined for the runway), not from the physical runway start, using a sign convention where positive values indicate distance past the threshold in the landing direction and negative values indicate touchdown prior to the threshold
3. THE Touchdown_Detector SHALL use geodesic or local tangent plane (ENU) coordinate math for projection, not naive Euclidean distance on raw latitude/longitude values, such that projection error does not exceed 0.1 m over a distance equal to the runway length
4. IF any of the required runway reference fields (threshold latitude, threshold longitude, runway heading, threshold elevation, or runway length) is missing, null, or outside valid bounds (latitude ±90, longitude ±180, heading 0–360, elevation −500 to 10000 m, length 0 to 6000 m) for a given flight, THEN THE Touchdown_Detector SHALL reject that flight with an error indication identifying which field is missing or invalid, and SHALL NOT produce a touchdown estimate for that flight
5. WHEN a runway has a displaced threshold, THE Touchdown_Detector SHALL use the displaced threshold coordinates as the reference origin, not the physical runway start coordinates, and the Along_Runway_Distance SHALL reflect the distance from that displaced threshold

### Requirement 12: Validation Against QAR Truth

**User Story:** As a safety analyst, I want rigorous validation against QAR ground truth, so that I can trust the system's accuracy claims and identify where it performs poorly.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL provide a validation harness that computes error metrics against QAR truth data (touchdown timestamp and lat/long position), reporting at minimum: signed distance error, absolute distance error, time error, RMSE, median absolute error, interquartile range, 95th-percentile absolute error, and 99th-percentile absolute error
2. THE Touchdown_Detector SHALL use Grouped_Splits (by aircraft tail number, airport, and runway) for train/test partitioning — no flight in the test set shall share a tail number, airport, or runway with any flight in the training set
3. THE Touchdown_Detector SHALL report signed Along_Runway_Distance error so that systematic long or short bias is visible, with positive error indicating the estimate is longer (farther from threshold) than truth
4. THE Touchdown_Detector SHALL report the full error distribution including median, interquartile range, 95th-percentile, and 99th-percentile absolute error, plus the 95th-percentile positive (long-side) signed error
5. THE Touchdown_Detector SHALL report error metrics stratified by aircraft type, ADS-B source, airport, and approach speed band, where each stratum contains at least 30 flights to be reportable
6. THE Touchdown_Detector SHALL compute a naive baseline (first on-ground sample position) and demonstrate that the system materially outperforms it, reporting both baseline and system metrics side-by-side
7. THE Touchdown_Detector SHALL evaluate cross-source generalization by training on one ADS-B source and testing on the other, reporting the same metrics as the primary validation

### Requirement 13: Accuracy Targets

**User Story:** As a safety analyst, I want the system to meet defined accuracy thresholds, so that estimates are precise enough for operational hazard identification.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL achieve an overall RMSE of Along_Runway_Distance error at or below 250 feet across all flights in the held-out test set when validated against QAR truth, where the test set contains a minimum of 4,000 flights and is split according to the grouped-split protocol defined in the validation requirements
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

1. WHEN given the same input data, configuration, random seeds, and software environment (Python version, library versions), THE Touchdown_Detector SHALL produce bit-identical outputs for all numeric fields in the output records
2. THE Touchdown_Detector SHALL propagate a single master random seed to all stochastic components (model initialization, data shuffling, dropout) such that setting one seed value determines all random behavior
3. THE Touchdown_Detector SHALL record with every output batch: data version identifier, git commit hash of the code, model artifact hash, resolved configuration (including defaults), Python version, and versions of key numerical libraries (NumPy, SciPy, scikit-learn, PyTorch/TensorFlow if used, LightGBM)

### Requirement 16: Derivative Signal Quality

**User Story:** As a data scientist, I want velocity derivatives (deceleration, jerk) computed with appropriate smoothing, so that the 4–5 second cadence does not produce noise-dominated signals.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL compute first-derivative (deceleration) and second-derivative (jerk) of groundspeed using either a Savitzky-Golay filter with polynomial order no greater than 3 and a window spanning at least 5 samples, or a Gaussian process, rather than naive finite differencing
2. THE Touchdown_Detector SHALL not use raw second-order finite differences (jerk) as a sole basis for determining `t_td`; jerk derived from unsmoothed finite differencing at 4–5 second cadence SHALL only be used as a corroborating signal alongside at least one other estimator
3. WHERE a Gaussian process is used for derivative estimation, THE Touchdown_Detector SHALL include the derivative posterior standard deviation in the estimator output alongside each derivative value, and downstream estimators SHALL weight the derivative inversely to its uncertainty
4. IF the number of valid groundspeed samples within the smoothing window is fewer than 5, THEN THE Touchdown_Detector SHALL flag the derivative as unreliable and emit a low-confidence indicator for any estimator that depends on that derivative
5. WHEN derivative quality is validated, THE Touchdown_Detector SHALL compare smoothed deceleration profiles against QAR-derived acceleration on a held-out sample and report the RMS discrepancy in m/s²

### Requirement 17: Vertical Profile Modeling

**User Story:** As a data scientist, I want the altitude-based estimator to model the flare correctly, so that the vertical crossing estimate is not biased long by fitting a straight line through a curved profile.

#### Acceptance Criteria

1. WHEN estimating touchdown via vertical profile crossing, THE Touchdown_Detector SHALL model the flare as a curved profile (exponential, quadratic, or piecewise-linear) rather than a single straight-line fit through the entire descent, where the flare region is defined as the segment below 50 feet above runway elevation
2. THE Touchdown_Detector SHALL use Geometric_Altitude (not Barometric_Altitude) for the vertical crossing calculation
3. WHEN the median Geometric_Altitude of on-approach samples deviates from the known runway elevation by more than 15 feet at the surface, THE Touchdown_Detector SHALL estimate and subtract the static bias before computing the crossing
4. THE Touchdown_Detector SHALL subtract the aircraft-type-specific antenna-to-main-gear height from the Geometric_Altitude (or equivalently add it to the crossing target elevation of runway_elevation) so that the crossing solution corresponds to main-gear contact rather than antenna height
5. IF fewer than 3 Geometric_Altitude samples exist within the flare region (below 50 feet above runway elevation), THEN THE Touchdown_Detector SHALL flag the vertical-profile estimate as low-confidence and fall back to an alternative estimator rather than fitting an under-constrained curve

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

1. THE Touchdown_Detector SHALL estimate the clock offset between QAR and ADS-B timestamps for each matched flight by aligning a common event observable in both data streams
2. THE Touchdown_Detector SHALL apply the estimated clock offset correction to QAR timestamps before using them as training labels or validation truth
3. THE Touchdown_Detector SHALL report the distribution of estimated clock offsets as a data quality diagnostic, including at minimum the median, standard deviation, and 95th-percentile absolute offset across the flight corpus
4. IF the estimated clock offset exceeds 2 seconds for a given flight, THEN THE Touchdown_Detector SHALL exclude that flight from model training and validation datasets and record it in a flagged-flights report for manual review
5. IF the clock offset cannot be reliably estimated for a given flight due to insufficient common-event quality, THEN THE Touchdown_Detector SHALL exclude that flight from model training and validation datasets and log the reason for estimation failure

### Requirement 20: Configuration-Driven Operation

**User Story:** As a data scientist, I want all tunable parameters externalized in configuration, so that experiments are reproducible and the system is adaptable without code changes.

#### Acceptance Criteria

1. THE Touchdown_Detector SHALL externalize the following parameters in configuration files: estimator selection and fusion weights, smoothing window sizes and polynomial orders, Gaussian process hyperparameters, Lever_Arm lookup values per aircraft type, plausibility thresholds, confidence thresholds, quality-gate limits (maximum gap size, minimum sample count), and validation split grouping definitions
2. THE Touchdown_Detector SHALL treat any numeric literal, threshold, window size, or lookup value that affects estimation output as a tunable parameter and SHALL NOT embed such values in source code
3. WHEN a configuration parameter is missing, THE Touchdown_Detector SHALL use a default value defined in a defaults section of the configuration schema and SHALL log a warning message identifying the parameter name and the default value applied
4. IF a configuration parameter value fails schema validation (wrong type, out of declared min/max range, or referencing an unknown estimator name), THEN THE Touchdown_Detector SHALL reject the configuration at startup, report an error message identifying the invalid parameter and the constraint violated, and SHALL NOT proceed with processing
5. WHEN a processing run completes, THE Touchdown_Detector SHALL record the complete resolved configuration (including any defaults applied) alongside the output so that the run can be reproduced with the identical parameter set
