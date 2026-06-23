"""Module 1: Ingest & QA.

Parse ADS-B, join runway/aircraft metadata, apply the per-source capability
descriptor, deduplicate, flag outliers via kinematic gates, and unify the
vertical datum (geoid-correct runway elevation to HAE).

Raw input schemas for the external ADS-B timeseries and QAR truth sources are
defined in :mod:`tdz.io.raw_schema` (distinct from the pipeline-internal models
in :mod:`tdz.models`).
"""

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
]
