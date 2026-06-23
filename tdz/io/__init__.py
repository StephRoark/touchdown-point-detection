"""Module 1: Ingest & QA.

Parse ADS-B, join runway/aircraft metadata, apply the per-source capability
descriptor, deduplicate, flag outliers via kinematic gates, and unify the
vertical datum (geoid-correct runway elevation to HAE).
"""
