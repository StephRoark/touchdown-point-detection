"""Module 1: Ingest & QA.

Parse ADS-B, join runway/aircraft metadata, apply the per-source capability
descriptor, deduplicate, flag outliers via kinematic gates, and unify the
vertical datum (geoid-correct runway elevation to HAE).

Raw input schemas for the external ADS-B timeseries and QAR truth sources are
defined in :mod:`tdz.io.raw_schema` (distinct from the pipeline-internal models
in :mod:`tdz.models`).

Public API
----------
Parsing (Task 9.1, :mod:`tdz.io.ingest`):

* :func:`parse_aireon_messages` -- Aireon async stream -> :class:`FlightRecord`
  with separate position/velocity timebases (Req 8.3).
* :func:`parse_fr24_records` -- co-timed FR24 rows -> :class:`FlightRecord` with
  ``position_times == velocity_times`` (Req 8.1, 8.2).
* :func:`find_qar_truth` -- light optional QAR-by-HEXID/registration lookup
  (``JOIN_KEYS``); the authoritative join is Task 21.
* :func:`aireon_messages_from_dataframe` / :func:`fr24_records_from_dataframe`
  -- optional pandas adapters (lazy import; the numpy-only core never needs
  pandas).

Source-capability gating (Task 9.2, :mod:`tdz.io.gating`; Req 8.5-8.8 /
Property 20):

* :func:`gate_estimators` / :class:`SourceGating` -- decide which estimators are
  eligible vs excluded (``GEOMETRIC_ALT_UNAVAILABLE``) from a
  :class:`~tdz.config.models.SourceCapability`, and surface the
  ``samples_independent`` flag for provider-interpolated sources.
* :data:`GEOMETRIC_ALTITUDE_ESTIMATORS` -- the geometric-altitude-dependent
  estimator ids.

File readers (:mod:`tdz.io.readers`) — delivered ADS-B/QAR files (CSV/Parquet)
-> pipeline-internal records:

* :func:`read_adsb_file` / :func:`read_adsb_flights` — load + schema-validate
  the delivered ADS-B timeseries and assemble per-flight
  :class:`~tdz.models.FlightRecord` objects (with :class:`AdsbFlightMeta` for
  the QAR join and tail-grouped split, and explicit :class:`SkippedFlight`
  reasons).
* :func:`read_runway_supplement` — the (airport, runway) geometry table for
  the fields the delivered files lack (heading, elevation+datum, width).
* :func:`read_qar_file` / :func:`match_qar_to_flights` — QAR truth rows and
  the airframe+landing-time join producing
  :class:`~tdz.models.QARTruthRecord` objects.
* :class:`RawUnits` — the explicit, flip-able unit assumptions for the
  'TODO: confirm' raw columns.

QA gates (Task 9.3, :mod:`tdz.io.qa`; Req 9.1, 9.3, 9.4, 9.5 / Properties 8, 9,
13):

* :func:`deduplicate_by_timestamp` -- last-received duplicate dedup (Property 8).
* :func:`apply_kinematic_gates` -- longitudinal/lateral/turn-rate exclusion
  (Property 9), using :data:`STANDARD_GRAVITY_MPS2`.
* :func:`evaluate_sufficiency` -- no-estimate reasons against a parameterized
  :class:`TouchdownWindow` (Req 9.5).
* :func:`run_qa` -- flight-level orchestration returning a :class:`QAResult`
  (cleaned record + :class:`QADiagnostics` + status/reason).
"""

from tdz.io.gating import (
    GEOMETRIC_ALTITUDE_ESTIMATORS,
    SourceGating,
    gate_estimators,
)
from tdz.io.ingest import (
    aireon_messages_from_dataframe,
    find_qar_truth,
    fr24_records_from_dataframe,
    parse_aireon_messages,
    parse_fr24_records,
)
from tdz.io.qa import (
    STANDARD_GRAVITY_MPS2,
    DedupResult,
    KinematicGateResult,
    QADiagnostics,
    QAResult,
    SufficiencyResult,
    TouchdownWindow,
    apply_kinematic_gates,
    deduplicate_by_timestamp,
    evaluate_sufficiency,
    run_qa,
)
from tdz.io.readers import (
    AdsbFlightMeta,
    AdsbReadResult,
    QARMatchResult,
    RawFileError,
    RawUnits,
    RunwaySupplementEntry,
    SkippedFlight,
    flight_records_from_adsb,
    match_qar_to_flights,
    qar_records_from_dataframe,
    read_adsb_file,
    read_adsb_flights,
    read_qar_file,
    read_runway_supplement,
)
from tdz.io.raw_schema import (
    ADSB_FIELDS,
    ADSB_TO_FLIGHTRECORD,
    JOIN_KEYS,
    QAR_FIELDS,
    QAR_TO_TRUTHRECORD,
    ADSBRecord,
    FieldSpec,
    QARRecord,
    adsb_columns,
    adsb_pandas_dtypes,
    qar_columns,
    qar_pandas_dtypes,
)

__all__ = [
    # raw_schema
    "FieldSpec",
    "ADSB_FIELDS",
    "QAR_FIELDS",
    "ADSBRecord",
    "QARRecord",
    "adsb_columns",
    "qar_columns",
    "adsb_pandas_dtypes",
    "qar_pandas_dtypes",
    "ADSB_TO_FLIGHTRECORD",
    "QAR_TO_TRUTHRECORD",
    "JOIN_KEYS",
    # ingest (Task 9.1)
    "parse_aireon_messages",
    "parse_fr24_records",
    "find_qar_truth",
    "aireon_messages_from_dataframe",
    "fr24_records_from_dataframe",
    # readers (delivered files)
    "RawFileError",
    "RawUnits",
    "RunwaySupplementEntry",
    "SkippedFlight",
    "AdsbFlightMeta",
    "AdsbReadResult",
    "QARMatchResult",
    "read_adsb_file",
    "read_adsb_flights",
    "read_qar_file",
    "read_runway_supplement",
    "flight_records_from_adsb",
    "qar_records_from_dataframe",
    "match_qar_to_flights",
    # gating (Task 9.2)
    "GEOMETRIC_ALTITUDE_ESTIMATORS",
    "SourceGating",
    "gate_estimators",
    # qa (Task 9.3)
    "STANDARD_GRAVITY_MPS2",
    "TouchdownWindow",
    "DedupResult",
    "KinematicGateResult",
    "SufficiencyResult",
    "QADiagnostics",
    "QAResult",
    "deduplicate_by_timestamp",
    "apply_kinematic_gates",
    "evaluate_sufficiency",
    "run_qa",
]
